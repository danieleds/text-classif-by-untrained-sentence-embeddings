import sys
sys.path.extend(['.', '..', '../..'])

from SMS.leaky_esn_attn.model import ESNModelSMS

import torch
from typing import Tuple, Any
import argparse
from datetime import datetime
from hyperopt import hp
from hyperopt.pyll.base import scope

from common import GridCellMeasurements, PerformanceMetrics, count_parameters
from common.Datasets.smsspam import SMSSpamDataset

import common.experiment
from common.experiment import ExtraArgs


argparser = argparse.ArgumentParser()
argparser.add_argument('--searches', type=int, default=60)
argparser.add_argument('--trials', type=int, default=3)
argparser.add_argument('--final-trials', type=int, default=1)
argparser.add_argument('--debug', action='store_true')
argparser.add_argument('--logname', type=str, default='')
argparser.add_argument('--epochs', type=int, help="Override the number of epochs")
argparser.add_argument('--saveto', type=str, default='')
args = argparser.parse_args()

train_fold, val_fold, test_fold = SMSSpamDataset.splits(root=common.get_cache_path())

extra = ExtraArgs(
    ds={
        'train': train_fold,
        'validation': val_fold,
        'test': test_fold
    }
)

# Hyperparameters from QC:
# opt_space = {'attention_r': 1, 'attention_type': 'LinSelfAttention', 'density_in': (0.6490642394189948,), 'density_in_bw': (0.6745601715844801,), 'dropout': 0.7970124251550051, 'epochs': 60, 'leaking_rate': (0.38930217546173795,), 'leaking_rate_bw': (0.7738524701335754,), 'lr': 0.000138326471408694, 'mlp_hidden_size': 0, 'mlp_n_hidden': 0, 'n_attention': 256, 'n_batch': 128, 'num_layers': 1, 'reservoir_size': 3000, 'scale_in': (6.745137376597286,), 'scale_in_bw': (0.2725147494411861,), 'scale_rec': (0.04641425166402264,), 'scale_rec_bw': (0.49182066901748145,), 'weight_decay': 0.0012623295223805023}

# Best hyperparameters:
opt_space = {'attention_r': 1, 'attention_type': 'LinSelfAttention', 'density_in': (0.1969143378291137,), 'density_in_bw': (0.19329875959229645,), 'dropout': 0.8445993209049081, 'epochs': 27, 'leaking_rate': (0.7202388559890865,), 'leaking_rate_bw': (0.8630659641792744,), 'lr': 0.00012880969397095887, 'mlp_hidden_size': 0, 'mlp_n_hidden': 0, 'n_attention': 128, 'n_batch': 128, 'num_layers': 1, 'reservoir_size': 1000, 'scale_in': (3.4990408845087195,), 'scale_in_bw': (8.92869426086838,), 'scale_rec': (0.001603604838365984,), 'scale_rec_bw': (0.1886769588948259,), 'weight_decay': 0.0023109597917704166}

def evaluate(hp: dict, extra: ExtraArgs, trial_id=0) -> Tuple[float, GridCellMeasurements]:
    """
    :return: a tuple composed of
      - a float specifying the score used for the random search (usually the validation accuracy)
      - a GridCellMeasurements object
    """
    if trial_id == 0:
        print(hp)

    train = extra.ds['train']
    val = extra.ds['validation']
    test = None

    if extra.is_final_trials:
        train = SMSSpamDataset.merge_folds([extra.ds['train'], val])
        val = None
        test = extra.ds['test']

    model = ESNModelSMS(hp, logname=args.logname)

    if trial_id == 0:
        print(f"# parameters: {count_parameters(model.model)}")

    model.fit(train, val)

    train_perf, val_perf, test_perf = model.performance(train, val, test)
    train_perf_f1, val_perf_f1, test_perf_f1 = model.performance_f1(train, val, test)
    train_perf_mcc, val_perf_mcc, test_perf_mcc = model.performance_mcc(train, val, test)

    if extra.is_final_trials:
        # Save the model
        datet = datetime.now().strftime('%b%d_%H-%M-%S')
        filename = f'SMS_leaky-esn-attn_{datet}_{trial_id}_{round(test_perf*100, 1)}.pt'
        torch.save(model.model.state_dict(), filename)

    metric_type = PerformanceMetrics.accuracy

    measurements = GridCellMeasurements(
        train_perf=train_perf,
        val_perf=val_perf,
        test_perf=test_perf,
        metric=metric_type.name,
        training_time=model.training_time,
        extra={
            metric_type.name: {
                'train': train_perf,
                'val': val_perf,
                'test': test_perf,
            },
            PerformanceMetrics.macro_f1.name: {
                'train': train_perf_f1,
                'val': val_perf_f1,
                'test': test_perf_f1
            },
            PerformanceMetrics.matthews_corrcoef.name: {
                'train': train_perf_mcc,
                'val': val_perf_mcc,
                'test': test_perf_mcc
            }
        },
        actual_epochs=model.actual_epochs
    )

    loss = 1/val_perf if val_perf > 0 else float('inf')
    return loss, measurements


if args.epochs is not None:
    opt_space['epochs'] = args.epochs

common.experiment.run_experiment(
    opt_space,
    evaluate,
    args.searches,
    args.trials,
    args.final_trials,
    extra
)
