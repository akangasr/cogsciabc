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

from cogsciabc.gridmodel.model import GridParams, get_model, get_dataset, discrepancy_function
from cogsciabc.gridmodel.mdp import InitialStateUniformlyAtEdge
from cogsciabc.log import logging_setup
from cogsciabc.args import parse_args_grid

import logging
logger = logging.getLogger(__name__)

def run_experiment(seed, method, grid_size, n_features, cores, samples):
    if n_features == 2:
        p = ModelParams([
            {"name": "feature1_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.1,
             "ntics": 0,
             },
            {"name": "feature2_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.1,
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
             "kernel_scale": 0.1,
             "ntics": 0,
             },
            {"name": "feature2_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.1,
             "ntics": 0,
             },
            {"name": "feature3_value",
             "distr": "uniform",
             "minv": -1.0,
             "maxv": 0.0,
             "acq_noise": 0.1,
             "kernel_scale": 0.1,
             "ntics": 0,
             },
            ])
    elfi_params = p.get_elfi_params()
    gp_params_update_interval = (cores-1)*2  # after every second batch
    parallel_batches = cores-1
    obs_set_size = 1000
    if grid_size < 12:
        path_max_len = 12  # limit to make exact method feasible
        sim_set_size = 2*obs_set_size
    else:
        path_max_len = None
        sim_set_size = obs_set_size
    training_eps = 2000 * grid_size

    if method in ["exact", "sample", "sample_l"]:
        noisy_posterior=True
        model_scale=-1000.0
    else:
        noisy_posterior=False
        model_scale=1.0

    rl_params = RLParams(
                n_training_episodes=training_eps,
                n_episodes_per_epoch=500,
                n_simulation_episodes=sim_set_size,
                q_alpha=0.2,
                q_w=0.5,
                q_gamma=0.99,
                q_iters=1,
                exp_epsilon=0.2,
                exp_decay=1.0)
    grid_params = GridParams(
                grid_size=grid_size,
                n_features=n_features,
                step_penalty=0.05,
                goal_value=float(grid_size),
                prob_rnd_move=0.05,
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
                noisy_posterior=noisy_posterior,
                model_scale=model_scale,
                n_samples=samples,
                n_initial_evidence=0,
                parallel_batches=parallel_batches,
                gp_params_optimizer="simplex",
                gp_params_max_opt_iters=20,
                gp_params_update_interval=gp_params_update_interval,
                batch_size=1,
                sampling_type="bo",
                seed=seed)

    if n_features == 2:
        ground_truth_v = [-0.33, -0.67]
        ground_truth = {"feature1_value": -0.33, "feature2_value": -0.67}
        training_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+1, path_max_len, obs_set_size)
        test_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+2, path_max_len, obs_set_size)
    if n_features == 3:
        ground_truth_v = [-0.25, -0.5, -0.75]
        ground_truth = {"feature1_value": -0.25, "feature2_value": -0.5, "feature3_value": -0.75}
        training_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+1, path_max_len, obs_set_size)
        test_data = get_dataset(grid_params, elfi_params, rl_params, ground_truth_v, seed+2, path_max_len, obs_set_size)

    # hack
    test_training_disc = discrepancy_function(InitialStateUniformlyAtEdge(grid_size), training_data, observed=[test_data])
    print("Discrepancy between test and training data was {:.4f}".format(test_training_disc[0]))

    if method in ["exact", "sample", "sample_l"]:
        types = ["MED", "LIK"]
        # hack
        from elfie.inference import SamplingPhase, PosteriorAnalysisPhase, PointEstimateSimulationPhase, PlottingPhase, GroundTruthErrorPhase, PredictionErrorPhase, LikelihoodSamplesSimulationPhase
        def modified_experiment(grid_params, elfi_params, rl_params, bolfi_params,
                                obs_data, test_data, plot_data, types, replicates, region_size,
                                ground_truth, n_cores, path_max_len, obs_set_size, pdf, figsize):
            elfi.new_model()
            model = get_model(method, grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len, obs_set_size)
            inference_task = BolfiFactory(model, bolfi_params).get()
            ret = dict()
            ret["n_cores"] = n_cores

            ret = SamplingPhase().run(inference_task, ret)
            ret = PosteriorAnalysisPhase(types=types).run(inference_task, ret)
            ret["plots_logl"] = inference_task.plot_post(pdf, figsize)

            elfi.new_model()
            model = get_model("approx", grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len, obs_set_size)
            inference_task = BolfiFactory(model, bolfi_params).get()

            ret = PointEstimateSimulationPhase(replicates=replicates, region_size=region_size).run(inference_task, ret)
            ret = LikelihoodSamplesSimulationPhase(replicates=replicates).run(inference_task, ret)
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
                      replicates=10,
                      region_size=0.02,
                      ground_truth=ground_truth,
                      n_cores=cores,
                      path_max_len=path_max_len,
                      obs_set_size=obs_set_size)
    if method in ["approx", "approx_l"]:
        types = ["ML", "LIK"]
        model = get_model(method, grid_params, elfi_params, rl_params, training_data, path_max_len, obs_set_size)
        inference_factory = BolfiFactory(model, bolfi_params)
        exp = partial(inference_experiment,
                      inference_factory,
                      test_data=test_data,
                      obs_data=training_data,
                      plot_data=None,
                      types=types,
                      ground_truth=ground_truth,
                      n_cores=cores,
                      replicates=10,
                      region_size=0.02)
    if method == "random":
        types = ["MD"]
        # hack
        from elfie.inference import PointEstimateSimulationPhase, PlottingPhase, GroundTruthErrorPhase, PredictionErrorPhase
        def modified_experiment(grid_params, elfi_params, rl_params, bolfi_params,
                                obs_data, test_data, plot_data, types, replicates, region_size,
                                ground_truth, n_cores, path_max_len, obs_set_size, seed, pdf, figsize):
            elfi.new_model()
            model = get_model("approx", grid_params, p.get_elfi_params(), rl_params, obs_data, path_max_len, obs_set_size)
            inference_task = BolfiFactory(model, bolfi_params).get()
            bounds = elfi_params.get_bounds()
            ret = dict()
            ret["n_cores"] = n_cores
            ret["MD"] = dict()
            random_state = np.random.RandomState(seed)
            for k, v in bounds.items():
                ret["MD"][k] = random_state.uniform(v[0], v[1])
            print("Random location: {}".format(ret["MD"]))
            ret["sampling_duration"] = 0
            ret["samples"] = dict()
            ret["n_samples"] = 0

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
                      replicates=10,
                      region_size=0.02,
                      ground_truth=ground_truth,
                      n_cores=cores,
                      path_max_len=path_max_len,
                      obs_set_size=obs_set_size,
                      seed=seed)

    file_path = os.path.dirname(os.path.realpath(__file__))
    run_and_report(exp, file_path)


if __name__ == "__main__":
    args = parse_args_grid()
    logging_setup()
    mpi_main(run_experiment, **args)

