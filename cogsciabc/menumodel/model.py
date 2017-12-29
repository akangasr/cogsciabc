from functools import partial
import numpy as np

import elfi
from elfie.serializable import Serializable
from elfirl.model import RLModel, RLParams
from cogsciabc.menumodel.mdp import SearchEnvironment, SearchTask

import logging
logger = logging.getLogger(__name__)

""" Menu search model.
    Chen et al. CHI 2016 and Kangasraasio et al. CHI 2017
"""

class MenuParams():

   def __init__(self,
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
                max_number_of_actions_per_session=20):
        d = {k: v for k, v in locals().items()}
        for k, v in d.items():
            setattr(self, k, v)

class Observation(Serializable):
    """ Summary observation: task completion in one of the possible scenarios:
        target absent or target present
    """
    def __init__(self, action_durations, target_present):
        super().__init__(self)
        self.task_completion_time = sum(action_durations)
        self.target_present = (target_present == True)

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.task_completion_time, self.target_present).__hash__()

    def __repr__(self):
        return "O({},{})".format(self.task_completion_time, self.target_present)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Observation(self.task_completion_time, self.target_present)


def get_model(p, elfi_p, rl_p, observation):
    env = SearchEnvironment(
                menu_type=p.menu_type,
                menu_groups=p.menu_groups,
                menu_items_per_group=p.menu_items_per_group,
                semantic_levels=p.semantic_levels,
                gap_between_items=p.gap_between_items,
                prop_target_absent=p.prop_target_absent,
                length_observations=p.length_observations,
                p_obs_len_cur=p.p_obs_len_cur,
                p_obs_len_adj=p.p_obs_len_adj,
                n_training_menus=p.n_training_menus)
    task = SearchTask(
                env=env,
                max_number_of_actions_per_session=p.max_number_of_actions_per_session)
    rl = RLModel(
                rl_params=rl_p,
                parameter_names=[p.name[4:] for p in elfi_p],
                env=env,
                task=task,
                clean_after_call=True)
    model = elfi_p[0].model
    simulator = elfi.Simulator(elfi.tools.vectorize(rl),
                               *elfi_p,
                               model=model,
                               observed=observation,
                               name="simulator")
    summary = elfi.Summary(elfi.tools.vectorize(partial(summary_function, maxlen=p.max_number_of_actions_per_session)),
                           simulator,
                           model=model,
                           name="summary")
    discrepancy = elfi.Discrepancy(elfi.tools.vectorize(discrepancy_function),
                                   summary,
                                   model=model,
                                   name="discrepancy")
    return model


def summary_function(obs, maxlen=None):
    while type(obs) is not dict:
        obs = obs[0]
    ret = list()
    lon_p = 0
    lon_a = 0
    neg_p = 0
    neg_a = 0
    for ses in obs["sessions"]:
        if maxlen is not None and len(ses["action_duration"]) >= maxlen:
            if ses["target_present"] is True:
                lon_p += 1
            else:
                lon_a += 1
            continue
        if "rewards" in ses.keys() and ses["rewards"][-1] < 0:
            if ses["target_present"] is True:
                neg_p += 1
            else:
                neg_a += 1
            continue
        ret.append(Observation(ses["action_duration"], ses["target_present"]))
    if len(ret) < len(obs["sessions"]):
        logger.info("Filtered observations, left {} out of {}".format(len(ret), len(obs["sessions"])))
        if lon_p + lon_a > 0:
            logger.info("Removed {} sessions that did not terminate (with {} actions or more), {} when target present, {} when absent".format(lon_p+lon_a, maxlen, lon_p, lon_a))
        if neg_p + neg_a > 0:
            logger.info("Removed {} sessions that were not done correctly (negative final reward), {} when target present, {} when absent".format(neg_p+neg_a, neg_p, neg_a))
    return ret


def discrepancy_function(*simulated, observed=None):
    while type(simulated[0]) is not Observation:
        simulated = simulated[0]
    while type(observed[0]) is not Observation:
        observed = observed[0]
    tct_mean_pre_obs, tct_std_pre_obs = _tct_mean_std(present=True, obs=simulated)
    tct_mean_pre_sim, tct_std_pre_sim = _tct_mean_std(present=True, obs=observed)
    tct_mean_abs_obs, tct_std_abs_obs = _tct_mean_std(present=False, obs=simulated)
    tct_mean_abs_sim, tct_std_abs_sim = _tct_mean_std(present=False, obs=observed)
    disc = np.abs(tct_mean_pre_obs - tct_mean_pre_sim) ** 2 \
            + np.abs(tct_std_pre_obs - tct_std_pre_sim) \
            + np.abs(tct_mean_abs_obs - tct_mean_abs_sim) ** 2 \
            + np.abs(tct_std_abs_obs - tct_std_abs_sim)
    disc = np.log(disc)
    return np.array([disc])


def _tct_mean_std(present, obs):
    tct = [o.task_completion_time for o in obs if o.target_present is present]
    if len(tct) == 0:
        logger.warning("No observations from condition: target present = {}".format(present))
        return 0.0, 0.0
    return np.mean(tct), np.std(tct)

