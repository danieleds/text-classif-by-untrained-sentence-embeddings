import sys
sys.path.extend(['.', '..', '../..'])

from SNLI.leaky_esn.model import ESNModelSNLI

import torch
from typing import Tuple, Any
import argparse
from datetime import datetime
from hyperopt import hp
from hyperopt.pyll.base import scope

from common import GridCellMeasurements, PerformanceMetrics, count_parameters
from common.Datasets.snli import SNLIDataset

import common.experiment
from common.experiment import ExtraArgs


argparser = argparse.ArgumentParser()
argparser.add_argument('--searches', type=int, default=60)
argparser.add_argument('--trials', type=int, default=3)
argparser.add_argument('--final-trials', type=int, default=1)
argparser.add_argument('--debug', action='store_true')
argparser.add_argument('--logname', type=str, default='')
argparser.add_argument('--saveto', type=str, default='')
args = argparser.parse_args()

train_fold, val_fold, test_fold = SNLIDataset.splits(root=common.get_cache_path())

extra = ExtraArgs(
    ds={
        'train': train_fold,
        'validation': val_fold,
        'test': test_fold
    }
)

# Hyperparameters from QC:
# opt_space = {'reservoir_size': 10000, 'num_layers': 1, 'density_in': [0.16241617337111502], 'density_in_bw': [0.8462420388657149], 'scale_in': [1.0662463087328995], 'scale_in_bw': [15.456222270037319], 'scale_rec': [2.2755404117929725], 'scale_rec_bw': [0.6080324164867185], 'r_alpha': 100, 'leaking_rate': [0.3202944730755585], 'leaking_rate_bw': [0.15651948678628058], 'n_batch': 128}

# Best hyperparameters:
opt_space = {'density_in': (0.9994663972644349,), 'density_in_bw': (0.9393418781794324,), 'leaking_rate': (0.22944333996542265,), 'leaking_rate_bw': (0.10989214430128433,), 'n_batch': 128, 'num_layers': 1, 'r_alpha': 0.025117260751334494, 'reservoir_size': 2500, 'scale_in': (4.5582937987309595,), 'scale_in_bw': (52.96784665383114,), 'scale_rec': (0.3773615469556043,), 'scale_rec_bw': (0.005511545875084299,)}

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
        #train = SNLIDataset.merge_folds([extra.ds['train'], val])
        val = None
        test = extra.ds['test']

    model = ESNModelSNLI(
        input_size=300,
        reservoir_size=hp['reservoir_size'],
        contractivity_coeff=hp['scale_rec'],
        scale_in=hp['scale_in'],
        leaking_rate=hp['leaking_rate'],
        alpha=hp['r_alpha'],
        rescaling_method='specrad',
        hp=hp
    )

    # Find the best value for the regularization parameter.
    # FIXME
    #if not extra.is_final_trials:
    #    best_alpha = model.find_best_alpha(train, val, hp['n_batch'])
    #    hp['r_alpha'] = best_alpha

    if trial_id == 0:
        print(f"# parameters: {6*hp['reservoir_size']}")

    model.fit(train)

    train_perf, val_perf, test_perf = model.performance(train, val, test)

    if extra.is_final_trials:
        # Save the model
        datet = datetime.now().strftime('%b%d_%H-%M-%S')
        filename = f'SNLI_leaky-esn_{datet}_{trial_id}_{round(test_perf*100, 1)}.pt'
        torch.save(model.state_dict(), filename)

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
            }
        }
    )

    loss = 1/val_perf if val_perf > 0 else float('inf')
    return loss, measurements


common.experiment.run_experiment(
    opt_space,
    evaluate,
    args.searches,
    args.trials,
    args.final_trials,
    extra
)
