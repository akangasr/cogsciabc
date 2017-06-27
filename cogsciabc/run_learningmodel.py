import os
import sys
from functools import partial

import matplotlib
matplotlib.use('Agg')

import numpy as np

import elfi
from elfie.bolfi_extensions import BolfiParams, BolfiFactory
from elfie.inference import inference_experiment, get_sample_pool
from elfie.mpi import mpi_main
from elfie.reporting import run_and_report
from elfie.params import ModelParams

from cogsciabc.learningmodel.model import LearningParams, get_model, get_dataset, plot_data
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed=1):
    p = ModelParams([
        {"name": "RT",
         "distr": "truncnorm",
         "minv": -5.0,
         "maxv": -2.5,
         "mean": -3.5,
         "std": 1.0,
         "acq_noise": 0.05,
         "kernel_scale": 2.0,
         "ntics": 20,
         },
        {"name": "LF",
         "distr": "truncnorm",
         "minv": 0.001,
         "maxv": 0.10,
         "mean": 0.05,
         "std": 0.10,
         "acq_noise": 0.001,
         "kernel_scale": 0.05,
         "ntics": 20,
         },
        ])
    training_data = get_dataset()
    model_params = LearningParams(
               sample_size=2,
               sample_d=0.01,
               max_retries=10,
               bounds=p.get_bounds(),
               )
    bolfi_params = BolfiParams(
                bounds=p.get_bounds(),
                grid_tics=p.get_grid_tics(),
                acq_noise_cov=p.get_acq_noises(),
                noise_var=0.5,
                kernel_var=10.0,
                kernel_scale=p.get_lengthscales(),
                ARD=True,
                n_samples=25,
                n_initial_evidence=5,
                parallel_batches=5,
                gp_params_update_interval=10,
                batch_size=1,
                sampling_type="bo",
#                pool=get_sample_pool("/m/home/home2/20/akangasr/unix/cogsciabc/cogsciabc/results2.json"),
                seed=args["seed"])

    model = get_model(model_params, p.get_elfi_params(), training_data)
    inference_factory = BolfiFactory(model, bolfi_params)

    file_path = os.path.dirname(os.path.realpath(__file__))
    exp = partial(inference_experiment,
                  inference_factory,
#                  skip_post=True,
                  obs_data=training_data,
                  test_data=training_data,
                  plot_data=plot_data)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

