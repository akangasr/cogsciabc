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

def run_experiment(seed, method, scale, cores, samples):
    logger.info("Running learning model with parameters")
    logger.info(" * seed = {}".format(seed))
    logger.info(" * method = {}".format(method))
    logger.info(" * scale = {}".format(scale))
    logger.info(" * cores = {}".format(cores))
    logger.info(" * samples = {}".format(samples))
    p = ModelParams([
        {"name": "RT",
         "distr": "truncnorm",
         "minv": -5.0,
         "maxv": -2.5,
         "mean": -3.5,
         "std": 1.0,
         "acq_noise": 0.05,
         "kernel_scale": 2.0,
         "L": 3.0,
         "ntics": scale,
         },
        {"name": "LF",
         "distr": "truncnorm",
         "minv": 0.001,
         "maxv": 0.10,
         "mean": 0.05,
         "std": 0.10,
         "acq_noise": 0.001,
         "kernel_scale": 0.05,
         "L": 30.0,
         "ntics": scale,
         },
        {"name": "BLC",
         "distr": "truncnorm",
         #"distr": "constant",
         #"val": 0.0,
         "minv": 0.0,
         "maxv": 20.0,
         "mean": 0.0,
         "std": 20.0,
         "acq_noise": 0.01,
         "kernel_scale": 1.0,
         "L": 0.5,
         "ntics": scale,
         },
        {"name": "ANS",
         "distr": "truncnorm",
         #"distr": "constant",
         #"val": 0.2,
         "minv": 0.001,
         "maxv": 0.3,
         "mean": 0.1,
         "std": 1.0,
         "acq_noise": 0.01,
         "kernel_scale": 1.0,
         "L": 20.0,
         "ntics": scale,
         }
        ])
    if method == "bo":
        gp_params_update_interval = 3*(cores-1)  # after every third batch
        skip_post = False
    else:
        gp_params_update_interval = 9999
        skip_post = True
    grid_tics = None
    if method == "grid":
        parallel_batches = samples
        grid_tics = p.get_grid_tics(seed)
    else:
        parallel_batches = cores-1
    training_data = get_dataset()
    model_params = LearningParams(max_retries=20)
    bolfi_params = BolfiParams(
                bounds=p.get_bounds(),
                grid_tics=grid_tics,
                acq_noise_cov=p.get_acq_noises(),
                noise_var=0.01,
                kernel_var=10.0,
                kernel_scale=p.get_lengthscales(),
                L=p.get_L(),
                ARD=True,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=parallel_batches,
                gp_params_update_interval=gp_params_update_interval,
                abc_threshold_delta=0.01,
                batch_size=1,
                sampling_type=method,
                seed=seed)

    model = get_model(model_params, p.get_elfi_params(), training_data)
    inference_factory = BolfiFactory(model, bolfi_params)

    file_path = os.path.dirname(os.path.realpath(__file__))
    exp = partial(inference_experiment,
                  inference_factory,
                  skip_post=skip_post,
                  obs_data=training_data,
                  test_data=training_data,
                  plot_data=plot_data,
                  n_cores=cores,
                  replicates=20,
                  region_size=0.02)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

