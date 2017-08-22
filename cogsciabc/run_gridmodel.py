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

from cogsciabc.gridmodel.model import GridParams, get_model, get_dataset
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args_grid

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed, exact, grid_size, n_features, cores, samples):
    if n_features == 2:
        p = ModelParams([
            {"name": "feature1_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.2,
             "L": 5.0,
             "ntics": 0,
             },
            {"name": "feature2_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.2,
             "L": 5.0,
             "ntics": 0,
             },
            ])
    if n_features == 3:
        p = ModelParams([
            {"name": "feature1_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.2,
             "L": 5.0,
             "ntics": 0,
             },
            {"name": "feature2_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.2,
             "L": 5.0,
             "ntics": 0,
             },
            {"name": "feature3_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.2,
             "L": 5.0,
             "ntics": 0,
             },
            ])
    elfi_params = p.get_elfi_params()
    gp_params_update_interval = cores-1
    parallel_batches = cores-1
    rl_params = RLParams(
                n_training_episodes=200000,
                n_episodes_per_epoch=100,
                n_simulation_episodes=1000,
                q_alpha=0.1,
                q_gamma=0.98,
                exp_epsilon=0.1,
                exp_decay=1.0)
    grid_params = GridParams(
                grid_size=grid_size,
                n_features=n_features,
                step_penalty=0.1,
                goal_value=10.0,
                prob_rnd_move=0.05,
                world_seed=seed,
                initial_state="edge",
                grid_type="walls",
                max_number_of_actions_per_session=100)
    bolfi_params = BolfiParams(
                bounds=p.get_bounds(),
                acq_noise_cov=p.get_acq_noises(),
                noise_var=0.1,
                kernel_var=10.0,
                kernel_scale=p.get_lengthscales(),
                L=p.get_L(),
                ARD=True,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=parallel_batches,
                gp_params_update_interval=gp_params_update_interval,
                batch_size=1,
                sampling_type="bo",
                seed=seed)

    path_max_len = 12
    if n_features == 2:
        ground_truth_v = [-0.33, -0.67]
        ground_truth = {"feature1_value": -0.33, "feature2_value": -0.67}
        training_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+1, max_sim_path_len=path_max_len)
        test_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+2, max_sim_path_len=path_max_len)
    if n_features == 3:
        ground_truth_v = [-0.25, -0.5, -0.75]
        ground_truth = {"feature1_value": -0.25, "feature2_value": -0.5, "feature3_value": -0.75}
        training_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+1, max_sim_path_len=path_max_len)
        test_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+2, max_sim_path_len=path_max_len)

    if exact is True:
        types = ["ML"]
        # hack
        from elfie.inference import SamplingPhase, PosteriorAnalysisPhase, PointEstimateSimulationPhase, PlottingPhase, GroundTruthErrorPhase, PredictionErrorPhase
        def modified_experiment(grid_params, elfi_params, rl_params, bolfi_params,
                                obs_data, test_data, plot_data, types, replicates, region_size,
                                ground_truth, n_cores, path_max_len, pdf, figsize):
            elfi.new_model()
            model = get_model(True, grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len)
            inference_task = BolfiFactory(model, bolfi_params).get()

            ret = SamplingPhase(n_cores=n_cores).run(inference_task, dict())
            ret = PosteriorAnalysisPhase(types=types).run(inference_task, ret)
            ret["plots_logl"] = inference_task.plot_post(pdf, figsize)

            elfi.new_model()
            model = get_model(False, grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len)
            inference_task = BolfiFactory(model, bolfi_params).get()

            ret = PointEstimateSimulationPhase(replicates=replicates, region_size=region_size).run(inference_task, ret)
            ret = PlottingPhase(pdf=pdf, figsize=figsize, obs_data=obs_data, test_data=test_data, plot_data=plot_data).run(inference_task, ret)
            ret = GroundTruthErrorPhase(ground_truth=ground_truth).run(inference_task, ret)
            ret = PredictionErrorPhase(test_data=test_data).run(inference_task, ret)
            return ret
        exp = partial(modified_experiment,
                      grid_params=grid_params,
                      elfi_params=p,
                      rl_params=rl_params,
                      bolfi_params=bolfi_params,
                      obs_data=training_data,
                      test_data=test_data,
                      plot_data=None,
                      types=types,
                      replicates=2,
                      region_size=0.01,
                      ground_truth=ground_truth,
                      n_cores=cores,
                      path_max_len=path_max_len)
    else:
        types = ["MED"]
        model = get_model(False, grid_params, elfi_params, rl_params, training_data, path_max_len)
        inference_factory = BolfiFactory(model, bolfi_params)
        exp = partial(inference_experiment,
                      inference_factory,
                      test_data=test_data,
                      obs_data=training_data,
                      plot_data=None,
                      types=types,
                      ground_truth=ground_truth,
                      n_cores=cores,
                      replicates=2,
                      region_size=0.01)

    file_path = os.path.dirname(os.path.realpath(__file__))
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args_grid()
    logging_setup()
    mpi_main(run_experiment, **args)

