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
         "minv": -2.60000001,
         "maxv": -2.6,
         "acq_noise": 0.0,
         "kernel_scale": 0.4,  # 20% of range
         "L": 2.5,  # 5 units / range
         "ntics": scale,
         },
        {"name": "LF",
         "distr": "truncnorm",
         "minv": 0.1,
         "maxv": 0.1000001,
         "mean": 0.2,
         "std": 0.2,
         "acq_noise": 0.0,
         "kernel_scale": 0.03,  # 20% of range
         "L": 33.3,  # 5 units / range
         "ntics": scale,
         },
        {"name": "BLC",
         "distr": "truncnorm",
         "minv": 2.0,
         "maxv": 2.0000001,
         "mean": 10.0,
         "std": 10.0,
         "acq_noise": 0.0,
         "kernel_scale": 4.0,  # 20% of range
         "L": 0.25,  # 5 units / range
         "ntics": scale,
         },
        {"name": "ANS",
         "distr": "truncnorm",
         "minv": 0.001,
         "maxv": 0.001000001,
         "mean": 0.3,
         "std": 0.2,
         "acq_noise": 0.0,
         "kernel_scale": 0.03,  # 20% of range
         "L": 33.3,  # 5 units / range
         "ntics": scale,
         }
        ])
    if method == "bo":
        gp_params_update_interval = min(samples, 50)
        types = ["MED", "MAP", "POST"]
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
                noise_var=0.0001,
                kernel_var=1.0,
                kernel_scale=p.get_lengthscales(),
                kernel_prior={"scale_E": 7.5, "scale_V": 37.5, "var_E": 2.0, "var_V": 2.0, "noise_E": 0.0004, "noise_V": 0.00000004},
                L=p.get_L(),
                ARD=True,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=parallel_batches,
                gp_params_update_interval=gp_params_update_interval,
                gp_params_optimizer="simplex",
                gp_params_max_opt_iters=100,
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
                  replicates=10,
                  region_size=0.02)
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args()
    logging_setup()
    mpi_main(run_experiment, **args)

