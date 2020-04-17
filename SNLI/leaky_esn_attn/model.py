import torch
from torch.utils.data import DataLoader
import time
import os
import math

import common
import common.embeddings
from common.base_models.attention import LinSelfAttention
from common.base_models.esn import ESNMultiringCell, ESNBase
from common.custom_models.leaky_esn_attention import LeakyESNAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(samples):
    return {
        'x1': torch.nn.utils.rnn.pad_sequence([ d['x1'] for d in samples ], batch_first=False),
        'x2': torch.nn.utils.rnn.pad_sequence([d['x2'] for d in samples], batch_first=False),
        'lengths1': [len(d['x1']) for d in samples],
        'lengths2': [len(d['x2']) for d in samples],
        'y': torch.cat([d['y'] for d in samples])
    }


def get_log_dir(comment=""):
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + comment)
    return os.path.join(os.path.realpath(__file__ + "/../"), log_dir)


class SNLIAttention(torch.nn.Module):
    # As shown here https://arxiv.org/pdf/1703.03130.pdf in Appendix B.

    def __init__(self, hidden_size, r):
        super(SNLIAttention, self).__init__()
        self.attn = LinSelfAttention(hidden_size, r=r)  # -> out: (batch, self.r * input_size), ...
        self.W_fh = torch.nn.Linear(r * hidden_size, hidden_size)
        self.W_fp = torch.nn.Linear(r * hidden_size, hidden_size)

    def forward(self, x1, x2):
        """

        Args:
            x1: tensor of shape (seq_len, batch, n_features)
            x2: tensor of shape (seq_len, batch, n_features)

        Returns:
            tensor of shape (batch, hidden_size)
        """
        M_h, _ = self.attn(x1)
        M_p, _ = self.attn(x2)
        F_h = self.W_fh(M_h)
        F_p = self.W_fp(M_p)
        return F_h * F_p


class ESNModelSNLI(torch.nn.Module):

    def __init__(self, hp, logname=''):
        super(ESNModelSNLI, self).__init__()

        input_size = 300
        output_size = 3
        self.epochs = hp['epochs']
        self.lr = hp['lr']
        self.batch_size = hp['n_batch']
        self.weight_decay = hp['weight_decay']
        self.n_layers = hp['num_layers']
        self.batch_size = hp['n_batch']
        self.reservoir_size = hp['reservoir_size']
        #self.dropout = hp['dropout']
        attention_hidden_size = hp['n_attention']
        attention_heads = hp['attention_r']

        num_directions = 2

        def cell_provider(input_size_, reservoir_size_, layer, direction):
            return ESNMultiringCell(
                input_size_,
                reservoir_size_,
                bias=True,
                contractivity_coeff=hp['scale_rec'][layer] if direction == 0 else hp['scale_rec_bw'][layer],
                scale_in=hp['scale_in'][layer] if direction == 0 else hp['scale_in_bw'][layer],
                density_in=hp['scale_in'][layer] if direction == 0 else hp['density_in_bw'][layer],
                leaking_rate=hp['leaking_rate'][layer] if direction == 0 else hp['leaking_rate_bw'][layer]
            )

        self.esn = ESNBase(
            cell_provider,
            input_size,
            self.reservoir_size,
            num_layers=self.n_layers,
            bidirectional=True
        ).to(device)

        # Dimensionality reduction
        self.ff1 = torch.nn.Linear(self.n_layers * num_directions * self.reservoir_size, attention_hidden_size).to(device)

        # Pairwise attention for SNLI
        self.attn = SNLIAttention(attention_hidden_size, r=attention_heads).to(device)

        # Classifier
        self.classifier = torch.nn.Linear(attention_hidden_size, output_size).to(device)

        self.early_stop = self.epochs < 0
        self.epochs = abs(self.epochs)

        self.training_time = -1
        self.actual_epochs = self.epochs

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        input: (seq_len, batch_size, input_size)
        output: (batch_size, N_Y)
        """

        s1, _ = self.esn.forward(x1.to(device))  # states: (seq_len, batch, num_directions * hidden_size)
        s1 = torch.tanh(self.ff1(s1))  # states: (seq_len, batch, n_attn)

        s2, _ = self.esn.forward(x2.to(device))  # states: (seq_len, batch, num_directions * hidden_size)
        s2 = torch.tanh(self.ff1(s2))  # states: (seq_len, batch, n_attn)

        # Apply Attention
        embedding = self.attn.forward(s1, s2)

        return self.classifier(embedding)

    def forward_in_batches(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=6)

        _Xs = []
        for _, minibatch in enumerate(dataloader):
            _Xs += [self.forward(minibatch['x1'].to(device, non_blocking=True),
                                 minibatch['x2'].to(device, non_blocking=True))]

        return torch.cat(_Xs, dim=0)  # FIXME Too slow

    def fit(self, train_fold, val_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param val_fold: validation fold.
        :return:
        """

        if self.early_stop and val_fold is None:
            raise Exception("User requested early stopping but a validation set was not provided")

        t_train_start = time.time()

        #weights = 1.0 / torch.Tensor([67, 945, 1004, 949, 670, 727])
        #sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_fold))
        #dataloader = DataLoader(train_fold, batch_size=self.batch_size, collate_fn=collate_fn,
        #                        pin_memory=True, sampler=sampler)

        dataloader = DataLoader(train_fold, shuffle=True, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        criterion = torch.nn.CrossEntropyLoss()

        checkpoint = self.state_dict()
        best_val_accuracy = 0
        epochs_without_val_acc_improvement = 0
        patience = 10
        epoch = 0
        #for epoch in tqdm(range(1, epochs + 1), desc="epochs", dynamic_ncols=True):
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            num_minibatches = 0
            for i, data in enumerate(dataloader):
                # Move data to devices
                data_x1 = data['x1'].to(device, non_blocking=True)
                data_x2 = data['x2'].to(device, non_blocking=True)
                data_y = data['y'].to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                self.train()

                train_out = self.forward(data_x1, data_x2)

                loss = criterion(train_out, data_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_minibatches += 1

            curr_avg_loss = running_loss/num_minibatches
            if math.isnan(curr_avg_loss):
                print("Loss is NaN. Stopping.")
                break

            if val_fold is not None:
                _, val_accuracy, _ = self.performance(None, val_fold, None)

                if val_accuracy > best_val_accuracy:
                    epochs_without_val_acc_improvement = 0
                    best_val_accuracy = val_accuracy
                    checkpoint = self.state_dict()
                else:
                    epochs_without_val_acc_improvement += 1

                # Early stopping
                if self.early_stop and epochs_without_val_acc_improvement >= patience:
                    print(f"Epoch {epoch}: no accuracy improvement after {patience} epochs. Early stop.")
                    self.load_state_dict(checkpoint)
                    break

        self.actual_epochs = epoch - patience if self.early_stop else epoch

        t_train_end = time.time()
        self.training_time = t_train_end - t_train_start

        # Compute accuracy on validation set
        _, val_accuracy, _ = self.performance(None, val_fold, None)
        return val_accuracy

    def performance(self, train_fold, val_fold, test_fold=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if train_fold:
            train_accuracy, train_out, train_expected = self.performance_from_fold(train_fold, batch_size)
        else:
            train_accuracy, train_out, train_expected = (0, None, None)

        if val_fold:
            val_accuracy, val_out, val_expected = self.performance_from_fold(val_fold, batch_size)
        else:
            val_accuracy, val_out, val_expected = (0, None, None)

        if test_fold:
            test_accuracy, test_out, test_expected = self.performance_from_fold(test_fold, batch_size)
        else:
            test_accuracy, test_out, test_expected = (0, None, None)

        save_raw_predictions = False
        if save_raw_predictions:
            raw_preds_filename = '/home/disarli/tmp/predictions.pt'
            try:
                saved = torch.load(raw_preds_filename)
            except FileNotFoundError:
                saved = []
            saved.append({
                'train_out': train_out.cpu(),
                'train_expected': train_expected.cpu(),
                'val_out': val_out.cpu(),
                'val_expected': val_expected.cpu(),
                'test_out': test_out.cpu() if test_fold else None,
                'test_expected': test_expected.cpu() if test_fold else None,
            })
            torch.save(saved, raw_preds_filename)

        return train_accuracy, val_accuracy, test_accuracy

    def performance_from_out(self, output, expected):
        """
        Given a tensor of network outputs and a tensor of expected outputs, returns the performance
        :param output:
        :param expected:
        :return:
        """
        output = output.argmax(dim=1).cpu()

        return common.accuracy(output, expected)

    def performance_from_fold(self, fold, batch_size):
        with torch.no_grad():
            self.eval()

            out = self.forward_in_batches(fold, batch_size)
            expected = torch.Tensor([d['y'] for d in fold])

            perf = self.performance_from_out(out, expected)
            return perf, out, expected
