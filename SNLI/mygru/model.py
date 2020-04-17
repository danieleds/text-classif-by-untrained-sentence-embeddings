import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import time
import scipy.io
from common.base_models.custom_rnn import CustomGRU
import math

import common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_trained_matrices = False


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


class MyGRUModelSNLI(torch.nn.Module):

    def __init__(self, hp, logname=''):
        super(MyGRUModelSNLI, self).__init__()

        input_size = 300
        reservoir_size = hp['reservoir_size']
        self.epochs = hp['epochs']
        self.lr = hp['lr']
        self.batch_size = hp['n_batch']
        self.weight_decay = hp['weight_decay']
        self.reservoir_size = reservoir_size

        num_directions = 2

        self.early_stop = self.epochs < 0
        self.epochs = abs(self.epochs)

        self.training_time = -1
        self.actual_epochs = self.epochs

        self.gru = CustomGRU(input_size, reservoir_size, bidirectional=(num_directions == 2)).to(device)
        self.readout = torch.nn.Linear(num_directions * reservoir_size * 3, 3).to(device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, seq_lengths_1=None, seq_lengths_2=None):
        """
        input: (seq_len, batch_size, input_size)
        output: (batch_size, N_Y)
        """

        s1, _ = self.gru(x1.to(device))
        # Extract last time step from each sequence
        s1 = self.gru.extract_last_time_step(s1, seq_lengths=seq_lengths_1)

        s2, _ = self.gru(x2.to(device))
        # Extract last time step from each sequence
        s2 = self.gru.extract_last_time_step(s2, seq_lengths=seq_lengths_2)

        concat_states = torch.cat([s1, s2, (s1-s2).abs()], dim=1)

        y_tilde = self.readout(concat_states)

        return y_tilde

    def fit(self, train_fold, val_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param val_fold: validation fold.
        :return:
        """

        t_train_start = time.time()

        epochs = self.epochs

        #weights = 1.0 / torch.Tensor([67, 945, 1004, 949, 670, 727])
        #sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_fold))
        #dataloader = DataLoader(train_fold, batch_size=self.batch_size, collate_fn=collate_fn,
        #                        pin_memory=True, sampler=sampler)

        dataloader = DataLoader(train_fold, shuffle=True, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=6)

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

                train_out = self.forward(data_x1, data_x2, seq_lengths_1=data['lengths1'], seq_lengths_2=data['lengths2'])

                loss = criterion(train_out, data_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_minibatches += 1

            curr_avg_loss = running_loss / num_minibatches
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

    def forward_in_batches(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=6)

        _Xs = []
        for _, minibatch in enumerate(dataloader):
            data_x1 = minibatch['x1'].to(device, non_blocking=True)
            data_x2 = minibatch['x2'].to(device, non_blocking=True)
            _Xs += [self.forward(data_x1, data_x2, seq_lengths_1=minibatch['lengths1'], seq_lengths_2=minibatch['lengths2'])]

        return torch.cat(_Xs, dim=0)

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
