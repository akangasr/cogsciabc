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
from elfirl.model import RLParams

from cogsciabc.menumodel.model import MenuParams, get_model, DataObject
from cogsciabc.menumodel.observation import BaillyData
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed=1):
    training_data = BaillyData(
                menu_type="Semantic",
                allowed_users=[],
                excluded_users=["S20", "S21", "S22", "S23", "S24"],
                trials_per_user_present=9999,  # all
                trials_per_user_absent=9999).get()  # all
    test_data = BaillyData(
                menu_type="Semantic",
                allowed_users=["S20", "S21", "S22", "S23", "S24"],
                excluded_users=[],
                trials_per_user_present=9999,  # all
                trials_per_user_absent=9999).get()  # all
    rl_params = RLParams(
                n_training_episodes=2000,
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
    elfi_params = [
                #elfi.Prior("truncnorm", -3, 3, 3, 1,
                elfi.Prior("uniform", 0, 6,
                #elfi.Constant(2.8,
                            name="focus_duration_100ms"),
                #elfi.Prior("truncnorm", -1, 0.7/0.3, 0.3, 0.3,  # TODO
                elfi.Constant(0.29,
                            name="selection_delay_s"),
                #elfi.Prior("truncnorm", -0.69/0.2, (1-0.69)/0.2, 0.69, 0.2,
                elfi.Constant(0.69,
                            name="menu_recall_probability"),
                #elfi.Prior("truncnorm", -0.93/0.2, (1-0.93)/0.2, 0.93, 0.2,
                elfi.Constant(0.93,
                            name="p_obs_adjacent")
                ]
    bolfi_params = BolfiParams(
                bounds=(
                    (0,6),
                    #(0,1),
                    #(0,1),
                    #(0,1)
                    ),
                n_samples=6,
                n_initial_evidence=2,
                parallel_batches=2,
                gp_params_update_interval=2,
                batch_size=1,
                sampling_type="uniform",
                seed=args["seed"])

    model = get_model(menu_params, elfi_params, rl_params, DataObject(training_data))
    inference_factory = BolfiFactory(model, bolfi_params)

    file_path = os.path.dirname(os.path.realpath(__file__))
    exp = partial(inference_experiment,
                  inference_factory,
                  test_data=DataObject(test_data))
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

