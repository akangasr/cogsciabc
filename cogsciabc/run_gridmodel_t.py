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

def run_experiment(seed, method, grid_size, n_features, cores, samples):
    assert n_features == 1
    assert cores == 2
    assert samples == 1
    p = ModelParams([
        {"name": "feature1_value",
         "distr": "uniform",
         "minv": 0.0,
         "maxv": 0.0,
         "acq_noise": 0.0,
         "kernel_scale": 0.1,
         "ntics": 0,
         },
        ])
    elfi_params = p.get_elfi_params()
    gp_params_update_interval = cores-1  # after every batch
    parallel_batches = cores-1
    path_max_len = 1000
    training_eps = 2000 * grid_size

    rl_params = RLParams(
                n_training_episodes=training_eps,
                n_episodes_per_epoch=500,
                n_simulation_episodes=200,
                q_alpha=0.2,
                q_w=0.5,
                q_gamma=0.99,
                q_iters=1,
                exp_epsilon=0.2,
                exp_decay=1.0)
    grid_params = GridParams(
                grid_size=grid_size,
                n_features=n_features,
                step_penalty=0.1,
                goal_value=float(grid_size),
                prob_rnd_move=0.1,
                world_seed=seed,
                initial_state="edge",
                grid_type="walls",
                max_number_of_actions_per_session=grid_size*10)
    bolfi_params = BolfiParams(
                bounds=p.get_bounds(),
                acq_noise_cov=p.get_acq_noises(),
                noise_var=0.1,
                kernel_var=10.0,
                kernel_scale=p.get_lengthscales(),
                kernel_prior={"scale_E": 0.1, "scale_V": 0.3, "var_E": 5.0, "var_V": 10.0},
                ARD=True,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=parallel_batches,
                gp_params_update_interval=gp_params_update_interval,
                batch_size=1,
                sampling_type="bo",
                seed=seed)

    ground_truth_v = [0.0]
    ground_truth = {"feature1_value": 0.0}
    training_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+1, max_sim_path_len=path_max_len)
    test_data = None

    if method in ["exact", "sample", "sample_l"]:
        types = ["MED"]
    if method == "approx":
        types = ["ML"]
    # hack
    from elfie.inference import SamplingPhase, PosteriorAnalysisPhase, PointEstimateSimulationPhase, PlottingPhase, GroundTruthErrorPhase, PredictionErrorPhase
    def modified_experiment(grid_params, elfi_params, rl_params, bolfi_params,
                            obs_data, test_data, plot_data, types, replicates, region_size,
                            ground_truth, n_cores, path_max_len, pdf, figsize):
        elfi.new_model()
        model = get_model(method, grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len)
        inference_task = BolfiFactory(model, bolfi_params).get()
        ret = dict()
        ret["n_cores"] = n_cores

        ret = SamplingPhase().run(inference_task, ret)
        ret = PosteriorAnalysisPhase(types=types).run(inference_task, ret)
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
                  replicates=0,
                  region_size=0.0,
                  ground_truth=ground_truth,
                  n_cores=cores,
                  path_max_len=path_max_len)

    file_path = os.path.dirname(os.path.realpath(__file__))
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args_grid()
    logging_setup()
    mpi_main(run_experiment, **args)

