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
         "distr": "uniform",
         "minv": -5.0,
         "maxv": -2.5,
         "acq_noise": 0.0,
         "kernel_scale": 0.5,  # 20% of range
         "L": 2.0,  # 5 units / range
         "ntics": scale,
         },
        {"name": "LF",
         "distr": "uniform",
         "minv": 0.001,
         "maxv": 0.10,
         "acq_noise": 0.0,
         "kernel_scale": 0.02,  # 20% of range
         "L": 50.0,  # 5 units / range
         "ntics": scale,
         },
        {"name": "BLC",
         "distr": "uniform",
         "minv": 0.0,
         "maxv": 20.0,
         "acq_noise": 0.0,
         "kernel_scale": 4.0,  # 20% of range
         "L": 0.25,  # 5 units / range
         "ntics": scale,
         },
        {"name": "ANS",
         "distr": "uniform",
         "minv": 0.001,
         "maxv": 0.10,
         "acq_noise": 0.0,
         "kernel_scale": 0.02,  # 20% of range
         "L": 50.0,  # 5 units / range
         "ntics": scale,
         }
        ])
    if method == "bo":
        gp_params_update_interval = 3*(cores-1)  # after every third batch
        types = ["MED", "ML"]  # uniform prior
    else:
        gp_params_update_interval = 9999
        types = ["MD"]
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
                noise_var=0.005,  # based on intial tests
                kernel_var=5.0,  # based on initial tests
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
                  obs_data=training_data,
                  test_data=training_data,
                  plot_data=plot_data,
                  types=types,
                  n_cores=cores,
                  replicates=20,
                  region_size=0.02)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

