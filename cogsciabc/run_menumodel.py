import os
import sys
from functools import partial

import matplotlib
matplotlib.use('Agg')

import numpy as np

import elfi
from elfie.bolfi_extensions import BolfiParams, BolfiFactory
from elfie.inference import inference_experiment
from elfie.mpi import mpi_main
from elfie.reporting import run_and_report
from elfie.params import ModelParams
from elfirl.model import RLParams

from cogsciabc.menumodel.model import MenuParams, get_model, summary_function
from cogsciabc.menumodel.observation import BaillyData
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed, method, scale, cores, samples):
    p = ModelParams([
        {"name": "focus_duration_100ms",
         "distr": "truncnorm",
         "minv": 0.0,
         "maxv": 6.0,
         "mean": 3.0,
         "std": 1.0,
         "acq_noise": 0.1,
         "kernel_scale": 1.0,
         "ntics": scale,
         },
        {"name": "selection_delay_s",
         #"distr": "constant",
         "val": 0.3,
         "distr": "truncnorm",
         "minv": 0.0,
         "maxv": 1.0,
         "mean": 0.3,
         "std": 0.3,
         "acq_noise": 0.05,
         "kernel_scale": 0.2,
         "ntics": scale,
         },
        {"name": "menu_recall_probability",
         "distr": "constant",
         "val": 0.69,
         #"distr": "truncnorm",
         "minv": 0.0,
         "maxv": 1.0,
         "mean": 0.69,
         "std": 0.2,
         "acq_noise": 0.05,
         "kernel_scale": 0.2,
         "ntics": scale,
         },
        {"name": "p_obs_adjacent",
         "distr": "constant",
         "val": 0.93,
         #"distr": "truncnorm",
         "minv": 0.0,
         "maxv": 1.0,
         "mean": 0.93,
         "std": 0.2,
         "acq_noise": 0.05,
         "kernel_scale": 0.2,
         "ntics": scale,
         },
        ])
    if method == "bo":
        gp_params_update_interval = cores-1
        skip_post = False
    else:
        gp_params_update_interval = 9999
        skip_post = True
    training_data = summary_function(BaillyData(
                menu_type="Semantic",
                allowed_users=[],
                excluded_users=["S20", "S21", "S22", "S23", "S24"],
                trials_per_user_present=9999,  # all
                trials_per_user_absent=9999).get())  # all
    test_data = summary_function(BaillyData(
                menu_type="Semantic",
                allowed_users=["S20", "S21", "S22", "S23", "S24"],
                excluded_users=[],
                trials_per_user_present=9999,  # all
                trials_per_user_absent=9999).get())  # all
    rl_params = RLParams(
                n_training_episodes=10000000,
                n_episodes_per_epoch=1000,
                n_simulation_episodes=10000,
                q_alpha=0.1,
                q_gamma=0.98,
                exp_epsilon=0.1,
                exp_decay=1.0)
    menu_params = MenuParams(
                menu_type="semantic",
                menu_groups=2,
                menu_items_per_group=4,
                semantic_levels=3,
                gap_between_items=0.75,
                prop_target_absent=0.1,
                length_observations=True,
                p_obs_len_cur=0.95,
                p_obs_len_adj=0.89,
                n_training_menus=10000,
                max_number_of_actions_per_session=20)
    bolfi_params = BolfiParams(
                bounds=p.get_bounds(),
                grid_tics=p.get_grid_tics(seed),
                acq_noise_cov=p.get_acq_noises(),
                noise_var=0.1,
                kernel_var=10.0,
                kernel_scale=p.get_lengthscales(),
                ARD=True,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=cores-1,
                gp_params_update_interval=gp_params_update_interval,
                batch_size=1,
                sampling_type=method,
                seed=seed)

    model = get_model(menu_params, p.get_elfi_params(), rl_params, training_data)
    inference_factory = BolfiFactory(model, bolfi_params)

    file_path = os.path.dirname(os.path.realpath(__file__))
    exp = partial(inference_experiment,
                  inference_factory,
                  skip_post=skip_post,
                  test_data=test_data,
                  obs_data=training_data,
                  plot_data=None,
                  n_cores=cores)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

