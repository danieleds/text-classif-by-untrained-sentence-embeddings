import sys
sys.path.extend(['.', '..', '../..'])

from SMS.leaky_esn_ensemble.model import ESNModelSMSEnsemble

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
# opt_space = {'reservoir_size': 10000, 'num_layers': 1, 'density_in': [0.16241617337111502], 'density_in_bw': [0.8462420388657149], 'scale_in': [1.0662463087328995], 'scale_in_bw': [15.456222270037319], 'scale_rec': [2.2755404117929725], 'scale_rec_bw': [0.6080324164867185], 'r_alpha': 100, 'leaking_rate': [0.3202944730755585], 'leaking_rate_bw': [0.15651948678628058], 'n_batch': 100, 'n_ensemble': 10}

# Best hyperparameters:
opt_space = {'density_in': (0.8566893332008322,), 'density_in_bw': (0.5396589010064617,), 'dropout': 0.20641405961275652, 'leaking_rate': (0.01754741029272547,), 'leaking_rate_bw': (0.4340210048855291,), 'n_batch': 128, 'num_layers': 1, 'r_alpha': 0.9244662151162362, 'reservoir_size': 5000, 'scale_in': (23.5027141847576,), 'scale_in_bw': (0.09230453394449425,), 'scale_rec': (0.03888164279426274,), 'scale_rec_bw': (0.22133187304071286,), 'n_ensemble': 10}
opt_space = {'density_in': (0.979579112327403,), 'density_in_bw': (0.5841900018321249,), 'dropout': 0.6865397273833402, 'leaking_rate': (0.23879665500567102,), 'leaking_rate_bw': (0.08660049406998288,), 'n_batch': 128, 'num_layers': 1, 'r_alpha': 0.0008613536159644131, 'reservoir_size': 10000, 'scale_in': (1.268067202671457,), 'scale_in_bw': (7.283686299742936,), 'scale_rec': (0.026586045793818507,), 'scale_rec_bw': (2.3052318562906335,), 'n_ensemble': 10}

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

    model = ESNModelSMSEnsemble(
        n_models=hp['n_ensemble'],
        input_size=300,
        reservoir_size=hp['reservoir_size'],
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
        print(f"# parameters: {6*hp['reservoir_size']*hp['n_ensemble']}")

    model.fit(train)

    train_perf, val_perf, test_perf = model.performance(train, val, test)
    train_perf_f1, val_perf_f1, test_perf_f1 = model.performance_f1(train, val, test)
    train_perf_mcc, val_perf_mcc, test_perf_mcc = model.performance_mcc(train, val, test)

    if extra.is_final_trials:
        # Save the model
        datet = datetime.now().strftime('%b%d_%H-%M-%S')
        filename = f'QC_leaky-esn-ensemble_{datet}_{trial_id}_{round(test_perf*100, 1)}.pt'
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
