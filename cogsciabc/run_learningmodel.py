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

from cogsciabc.learningmodel.model import LearningParams, get_model, get_dataset, DataObject
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed=1):
    training_data = get_dataset()
    model_params = LearningParams(
               sample_size=6,
               sample_d=0.005,
               max_retries=5,
               )
    elfi_params = [
                elfi.Prior("uniform", -3.0, 0.7, name="a"),
                elfi.Prior("uniform", 0.05, 0.1, name="b"),
                ]
    bolfi_params = BolfiParams(
                bounds=(
                    (-3.0,-2.3),
                    (0.05,0.15),
                    #(0,1),
                    #(0,1)
                    ),
                n_samples=100,
                n_initial_evidence=10,
                parallel_batches=4,
                gp_params_update_interval=4,
                batch_size=1,
                sampling_type="uniform",
                seed=args["seed"])

    model = get_model(model_params, elfi_params, DataObject(training_data))
    inference_factory = BolfiFactory(model, bolfi_params)

    file_path = os.path.dirname(os.path.realpath(__file__))
    exp = partial(inference_experiment,
                  inference_factory)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

