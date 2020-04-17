import torch
from torch.utils.data import DataLoader
from common.base_models.esn import ESNBase, ESNMultiringCell
import time
from sklearn.linear_model import RidgeClassifier
from common.base_models.big_ridge import BigRidgeClassifier
from tqdm import tqdm

import common
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(samples):
    return {
        'x1': torch.nn.utils.rnn.pad_sequence([ d['x1'] for d in samples ], batch_first=False),
        'x2': torch.nn.utils.rnn.pad_sequence([d['x2'] for d in samples], batch_first=False),
        'lengths1': [len(d['x1']) for d in samples],
        'lengths2': [len(d['x2']) for d in samples],
        'y': torch.cat([d['y'] for d in samples])
    }


class ESNModelSNLI(torch.nn.Module):

    def __init__(self, input_size, reservoir_size, contractivity_coeff=0.9,
                 scale_in=1.0, leaking_rate=1.0, alpha=1e-6, rescaling_method='norm',
                 hp=None):
        super(ESNModelSNLI, self).__init__()

        bidirectional = True

        self.n_layers = hp['num_layers']
        self.batch_size = hp['n_batch']

        def cell_provider(input_size_, reservoir_size_, layer, direction):
            return ESNMultiringCell(
                input_size_,
                reservoir_size_,
                bias=True,
                contractivity_coeff=hp['scale_rec'][layer] if direction == 0 else hp['scale_rec_bw'][layer],
                scale_in=hp['scale_in'][layer] if direction == 0 else hp['scale_in_bw'][layer],
                density_in=hp['scale_in'][layer] if direction == 0 else hp['density_in_bw'][layer],
                leaking_rate=hp['leaking_rate'][layer] if direction == 0 else hp['leaking_rate_bw'][layer],
                rescaling_method=rescaling_method
            )

        self.esn = ESNBase(
            cell_provider,
            input_size,
            reservoir_size,
            num_layers=self.n_layers,
            bidirectional=bidirectional
        ).to(device)

        # self.regr = RidgeClassifier(alpha=alpha, class_weight='balanced', normalize=True)
        self.regr = None
        self.alpha = alpha

        # Cached output for the training set
        self.cached_train_out = None

        self.training_time = -1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, seq_lengths_1=None, seq_lengths_2=None, return_states=False, return_probs=False):
        """
        input: (seq_len, batch_size, input_size)
        lengths: integer list of lengths, one for each sequence in 'input'. If provided, padding states
                 are automatically ignored.
        output: (1,)
        """

        # x: (seq_len, batch, num_directions * hidden_size)      # last layer only, all steps
        # h: (num_layers * num_directions, batch, hidden_size)   # last step only, all layers
        x, h = self.esn(x1.to(device))

        if self.n_layers == 1:
            state = self.esn.extract_last_states(x, seq_lengths=seq_lengths_1).cpu()
            s1 = state  # Tensor( batch_size, num_direction * output_size * n_layers=1 )
        else:
            # Tensor( batch_size, num_direction * output_size * n_layers )
            s1 = h.permute(1, 0, 2).contiguous().view(h.shape[1], -1).cpu()

        x, h = self.esn(x2.to(device))
        if self.n_layers == 1:
            state = self.esn.extract_last_states(x, seq_lengths=seq_lengths_2).cpu()
            s2 = state  # Tensor( batch_size, num_direction * output_size * n_layers=1 )
        else:
            # Tensor( batch_size, num_direction * output_size * n_layers )
            s2 = h.permute(1, 0, 2).contiguous().view(h.shape[1], -1).cpu()

        concat_states = torch.cat([s1, s2, (s1-s2).abs()], dim=1)

        if return_states:
            return concat_states

        if return_probs:
            return torch.from_numpy(self.regr.decision_function(concat_states.numpy()))
        else:
            return torch.from_numpy(self.regr.predict(concat_states.numpy()))

    def forward_in_batches(self, dataset, batch_size, return_states=False, return_probs=False):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=6)

        _Xs = []
        for _, minibatch in enumerate(dataloader):
            _Xs += [self.forward(minibatch['x1'].to(device, non_blocking=True),
                                 minibatch['x2'].to(device, non_blocking=True),
                                 seq_lengths_1=minibatch['lengths1'],
                                 seq_lengths_2=minibatch['lengths2'],
                                 return_states=return_states,
                                 return_probs=return_probs)]

        return torch.cat(_Xs, dim=0)  # FIXME Too slow

    def find_best_alpha(self, train_fold, val_fold, batch_size):
        pass

    def fit(self, train_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param batch_size:
        :return:
        """
        self.train()

        # Collect all states
        t_train_start = time.time()

        _X = self.forward_in_batches(train_fold, self.batch_size, return_states=True)
        _y = torch.cat([d['y'] for d in train_fold]).to(device).type(torch.get_default_dtype())

        _X = _X.cpu().numpy()
        _y = _y.cpu().numpy()

        # Actual training
        self._fit_from_states(_X, _y)

        t_train_end = time.time()

        self.training_time = t_train_end - t_train_start

        self.cached_train_out = torch.from_numpy(self.regr.predict(_X))

    def _fit_from_states(self, states, expected):
        """
        Fit the model from a matrix of states
        :param states: a tensor or numpy array of shape ( batch_size, state_size )
        :param expected: expected index for each sample
        :return:
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(expected, torch.Tensor):
            expected = expected.cpu().numpy()

        self.regr = RidgeClassifier(alpha=self.alpha, normalize=False)
        #self.regr = BigRidgeClassifier(alpha=self.alpha)
        self.regr.fit(states, expected)
        # W = (X.t() @ X + (self.alpha)*torch.eye(X.shape[1], device=device)).inverse() @ X.t() @ y

    def _predict_from_states(self, states):
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        return torch.from_numpy(self.regr.predict(states))

    def performance(self, train_fold, val_fold, test_fold=None):
        with torch.no_grad():

            # Actual output class indices
            self.eval()
            train_out = self.cached_train_out
            val_out = self.forward_in_batches(val_fold, self.batch_size) if val_fold else None
            test_out = self.forward_in_batches(test_fold, self.batch_size) if test_fold else None

            # Expected output class indices
            train_expected = torch.as_tensor([ d['y'] for d in train_fold ])
            val_expected = torch.as_tensor([ d['y'] for d in val_fold ]) if val_fold else None
            test_expected = torch.as_tensor([ d['y'] for d in test_fold ]) if test_fold else None

            save_raw_predictions = False
            if save_raw_predictions:
                raw_preds_filename = '/home/disarli/tmp/predictions.pt'
                raw_train_out = self.forward_in_batches(train_fold, self.batch_size, return_probs=True)
                raw_val_out = self.forward_in_batches(val_fold, self.batch_size, return_probs=True) if val_fold else None
                raw_test_out = self.forward_in_batches(test_fold, self.batch_size, return_probs=True) if test_fold else None
                try:
                    saved = torch.load(raw_preds_filename)
                except FileNotFoundError:
                    saved = []
                saved.append({
                    'train_out': raw_train_out.cpu(),
                    'train_expected': train_expected.cpu(),
                    'val_out': raw_val_out.cpu() if val_fold else None,
                    'val_expected': val_expected.cpu() if val_fold else None,
                    'test_out': raw_test_out.cpu() if test_fold else None,
                    'test_expected': test_expected.cpu() if test_fold else None,
                })
                torch.save(saved, raw_preds_filename)

            # Compute performance measures
            train_accuracy = common.accuracy(train_out, train_expected)
            val_accuracy = common.accuracy(val_out, val_expected) if val_fold else 0
            test_accuracy = common.accuracy(test_out, test_expected) if test_fold else 0

            return train_accuracy, val_accuracy, test_accuracy

    @staticmethod
    def ensemble_performance(predictions, expected):
        out = torch.stack(predictions).mean(dim=0)
        out = out.argmax(dim=1)
        return common.accuracy(out, expected)


class ESNModelSNLIEnsemble(torch.nn.Module):

    def __init__(self, n_models, input_size, reservoir_size, alpha=1e-6,
                 rescaling_method='norm', hp=None):
        super(ESNModelSNLIEnsemble, self).__init__()

        self.models = [ ESNModelSNLI(
            input_size,
            reservoir_size,
            alpha=alpha,
            rescaling_method=rescaling_method,
            hp=hp
        ) for _ in range(n_models) ]

        self.batch_size = hp['n_batch']

        self.training_time = -1

    def find_best_alpha(self, train_fold, val_fold):
        """
        Fit the model while searching for the best regularization parameter. The best regularization
        parameter is then assigned to self.alpha.
        :param train_fold:
        :param val_fold:
        :param batch_size:
        :return:
        """

        self.models[0].find_best_alpha(train_fold, val_fold, self.batch_size)
        best_alpha = self.models[0].alpha

        for m in self.models:
            m.alpha = best_alpha

        return best_alpha

    def fit(self, train_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param batch_size:
        :return:
        """

        for m in self.models:
            m.fit(train_fold, self.batch_size)

        self.training_time = sum([m.training_time for m in self.models])

    def performance(self, train_fold, val_fold, test_fold=None):
        with torch.no_grad():

            # Expected output class indices
            train_expected = torch.Tensor([ d['y'] for d in train_fold ])
            val_expected = torch.Tensor([ d['y'] for d in val_fold ])
            test_expected = torch.Tensor([ d['y'] for d in test_fold ]) if test_fold else None

            for m in self.models:
                m.eval()

            train_outs = [ m.forward_in_batches(train_fold, self.batch_size, return_probs=True) for m in self.models ]
            val_outs = [ m.forward_in_batches(val_fold, self.batch_size, return_probs=True) for m in self.models ]
            test_outs = [ m.forward_in_batches(test_fold, self.batch_size, return_probs=True) for m in self.models ] if test_fold else None

            train_out = torch.stack(train_outs).mean(dim=0).argmax(dim=1)
            val_out = torch.stack(val_outs).mean(dim=0).argmax(dim=1)
            test_out = torch.stack(test_outs).mean(dim=0).argmax(dim=1) if test_fold else None

            # Compute performance measures
            train_accuracy = common.accuracy(train_out, train_expected)
            val_accuracy = common.accuracy(val_out, val_expected)
            test_accuracy = common.accuracy(test_out, test_expected) if test_fold else 0

            return train_accuracy, val_accuracy, test_accuracy
