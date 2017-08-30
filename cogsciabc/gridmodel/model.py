from functools import partial
import time
import numpy as np

import elfi
from elfie.serializable import Serializable
from elfirl.model import RLModel, RLParams
from cogsciabc.gridmodel.mdp import *

import logging
logger = logging.getLogger(__name__)

class GridParams():

   def __init__(self,
                grid_size=11,
                n_features=2,
                step_penalty=0.1,
                goal_value=1.0,
                prob_rnd_move=0.1,
                world_seed=0,
                p_grid_feature=0.4,
                path_max_len=1e10,
                initial_state="edge",
                grid_type="walls",
                max_number_of_actions_per_session=20):
        d = {k: v for k, v in locals().items()}
        for k, v in d.items():
            setattr(self, k, v)

class Observation(dict):
    """ Summary observation: start state of path and path length
    """
    def __init__(self, path=None, start_state=None, path_len=None):
        if path is not None:
            self.start_state_x = int(path.transitions[0].prev_state.x)
            self.start_state_y = int(path.transitions[0].prev_state.y)
            self.path_len = len(path)
        else:
            self.start_state_x = int(start_state.x)
            self.start_state_y = int(start_state.y)
            self.path_len = int(path_len)

    @property
    def start_state(self):
        return State(self.start_state_x, self.start_state_y)

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.start_state_x, self.start_state_y, self.path_len).__hash__()

    def __repr__(self):
        return "O(({},{}),{})".format(self.start_state_x, self.start_state_y, self.path_len)

    def __str__(self):
        return self.__repr__()


class PathTreeIterator():
    """ Iterator for a path tree

    Parameters
    ----------
    root : Observation
    paths : dict[observation] = node with nodes being list of tuples (transition, next observation)
    """
    def __init__(self, root, paths, maxlen):
        self.root = root
        self.paths = paths
        self.maxlen = maxlen

    def __iter__(self):
        self.indices = [0] * self.maxlen
        self.end = (len(self.paths[self.root]) == 0)
        return self

    def __next__(self):
        while True:
            if self.end is True:
                raise StopIteration()
            path = None
            nvals = list()
            try:
                path = Path([])
                node = self.paths[self.root]
                for i in self.indices:
                    assert len(node) > 0
                    nvals.append(len(node))
                    transition, next_obs = node[i]
                    if transition is None and next_obs is None:
                        # dead end
                        path = None
                        raise IndexError
                    path.append(transition)
                    assert next_obs in self.paths, "Observation {} not found in tree?".format(next_obs)
                    node = self.paths[next_obs]
            except IndexError:
                # dead end
                pass
            for i in reversed(range(len(nvals))):
                if self.indices[i] < nvals[i]-1:
                    self.indices[i] += 1
                    break
                else:
                    self.indices[i] = 0
            if max(self.indices) == 0:
                self.end = True
            if path is not None:
                return path

def get_dataset(p, elfi_p, rl_p, param_values, seed, max_sim_path_len=1e10):
    logger.info("Generating a dataset with parameters {}..".format(param_values))
    if p.initial_state == "edge":
        initial_state_generator = InitialStateUniformlyAtEdge(p.grid_size)
    elif p.initial_state == "anywhere":
        initial_state_generator = InitialStateUniformlyAnywhere(p.grid_size)
    else:
        raise ValueError("Unknown initial state type: {}".format(p.initial_state))

    if p.grid_type == "uniform":
        grid_generator = UniformGrid(p.world_seed, p_feature=p.p_grid_feature)
    elif p.grid_type == "walls":
        grid_generator = WallsGrid(p.world_seed, n_walls_per_feature=p.grid_size)
    else:
        raise ValueError("Unknown grid type: {}".format(p.grid_type))

    goal_state = State(int(p.grid_size/2), int(p.grid_size/2))
    env = GridEnvironment(
                grid_size=p.grid_size,
                prob_rnd_move=p.prob_rnd_move,
                n_features=p.n_features,
                goal_state=goal_state,
                initial_state_generator=initial_state_generator,
                grid_generator=grid_generator)
    task = GridTask(
                env=env,
                step_penalty=p.step_penalty,
                goal_value=p.goal_value,
                max_number_of_actions_per_session=p.max_number_of_actions_per_session)
    rl = RLModel(
                rl_params=rl_p,
                parameter_names=[p.name[4:] for p in elfi_p],
                env=env,
                task=task,
                clean_after_call=False)
    data = rl(*param_values, random_state=np.random.RandomState(seed))
    policy = rl.get_policy()
    rl.env.print_policy(policy)
    summary = filt_summary(max_sim_path_len, data)
    logger.info("Dataset generated")
    print(summary)
    return summary

def get_model(variant, p, elfi_p, rl_p, observation, path_max_len):
    if p.initial_state == "edge":
        initial_state_generator = InitialStateUniformlyAtEdge(p.grid_size)
    elif p.initial_state == "anywhere":
        initial_state_generator = InitialStateUniformlyAnywhere(p.grid_size)
    else:
        raise ValueError("Unknown initial state type: {}".format(p.initial_state))

    if p.grid_type == "uniform":
        grid_generator = UniformGrid(p.world_seed, p_feature=p.p_grid_feature)
    elif p.grid_type == "walls":
        grid_generator = WallsGrid(p.world_seed, n_walls_per_feature=p.grid_size)
    else:
        raise ValueError("Unknown grid type: {}".format(p.grid_type))

    goal_state = State(int(p.grid_size/2), int(p.grid_size/2))
    env = GridEnvironment(
                grid_size=p.grid_size,
                prob_rnd_move=p.prob_rnd_move,
                n_features=p.n_features,
                goal_state=goal_state,
                initial_state_generator=initial_state_generator,
                grid_generator=grid_generator)
    task = GridTask(
                env=env,
                step_penalty=p.step_penalty,
                goal_value=p.goal_value,
                max_number_of_actions_per_session=p.max_number_of_actions_per_session)
    rl = RLModel(
                rl_params=rl_p,
                parameter_names=[p.name[4:] for p in elfi_p],
                env=env,
                task=task,
                clean_after_call=False)
    if variant == "exact":
        simulator = elfi.Simulator(elfi.tools.vectorize(dummy_simulator),
                                   *elfi_p,
                                   name="simulator")
        summary = elfi.Summary(elfi.tools.vectorize(partial(passthrough_summary_function, path_max_len)),
                               simulator,
                               observed=observation,
                               name="summary")
        discrepancy = elfi.Discrepancy(elfi.tools.vectorize(partial(logl_discrepancy,
                                                                    rl, p.max_number_of_actions_per_session, goal_state,
                                                                    env.transition_prob, full=True)),
                                       summary,
                                       name="discrepancy")
    if variant == "sample":
        simulator = elfi.Simulator(elfi.tools.vectorize(dummy_simulator),
                                   *elfi_p,
                                   name="simulator")
        summary = elfi.Summary(elfi.tools.vectorize(partial(passthrough_summary_function, path_max_len)),
                               simulator,
                               observed=observation,
                               name="summary")
        discrepancy = elfi.Discrepancy(elfi.tools.vectorize(partial(logl_discrepancy,
                                                                    rl, p.max_number_of_actions_per_session, goal_state,
                                                                    env.transition_prob, full=False)),
                                       summary,
                                       name="discrepancy")

    if variant == "approx":
        simulator = elfi.Simulator(elfi.tools.vectorize(rl),
                                   *elfi_p,
                                   name="simulator")
        summary = elfi.Summary(elfi.tools.vectorize(partial(filt_summary, path_max_len)),
                               simulator,
                               observed=observation,
                               name="summary")
        discrepancy = elfi.Discrepancy(elfi.tools.vectorize(partial(discrepancy_function, initial_state_generator)),
                                       summary,
                                       name="discrepancy")
    return elfi.get_current_model()


def simulator(rl, *parameters, index_in_batch=0, random_state=None):
    ret = rl(*parameters, random_state=random_state)
    policy = rl.get_policy()
    rl.env.print_policy(policy)
    return ret

def filt_summary(path_max_len, observations):
    obs = summary_function(observations)
    filt_obs = [o for o in obs if o.path_len <= path_max_len]
    if len(filt_obs) < len(obs):
        logger.info("Filtered observations to be at most length {}, left {} out of {}"\
                .format(path_max_len, len(filt_obs), len(obs)))
    return filt_obs

def summary_function(observations):
    obs = [Observation(ses["path"]) for ses in observations["sessions"]]
    return obs

class DummyValue():
    """ Used to pass values from simulator to discrepancy function for
        computing the likelihood of the observations
    """
    def __init__(self, parameters, random_state):
        self.parameters = parameters
        self.random_state = random_state

# Exact inference using logl
def dummy_simulator(*parameters, index_in_batch=0, random_state=None):
    return DummyValue(parameters, random_state)

def passthrough_summary_function(path_max_len, data):
    if isinstance(data, DummyValue):
        return data
    else:
        return filt_summary(path_max_len, data)

def logl_discrepancy(rl, path_max_len, goal_state, transition_prob, *sim_data, observed=None, full=True):
    parameters = sim_data[0].parameters
    random_state = sim_data[0].random_state
    return np.array([-1 * evaluate_loglikelihood(rl, path_max_len, goal_state, transition_prob,
                                                 parameters, observed[0], random_state, full=full)])

def evaluate_loglikelihood(rl, path_max_len, goal_state, transition_prob,
                           parameters, observations, random_state, scale=100.0, full=True):
    # Note: scaling != 1.0 will not preserve proportionality of likelihood
    # (only used as a hack to make GP implementation work, as it breaks with too large values)
    assert len(observations) > 0
    ind_log_obs_probs = list()
    rl.train_model(parameters, random_state=random_state)
    precomp_obs_logprobs = dict()
    policy = rl.get_policy()
    rl.env.print_policy(policy)
    if full is not True:
        sims = rl.simulate(random_state=random_state)
        sim_paths = [ses["path"] for ses in sims["sessions"]]
    #logger.info("Evaluating loglikelihood of {} observations..".format(len(observations)))
    start_time1 = time.time()
    for obs_i in observations:
        if obs_i in precomp_obs_logprobs.keys():
            #logger.info("Using precomputed loglikelihood of {}".format(obs_i))
            logprob = precomp_obs_logprobs[obs_i]
            ind_log_obs_probs.append(logprob)
            continue
        #logger.info("Evaluating loglikelihood of {}..".format(obs_i))
        start_time2 = time.time()
        n_paths = 0
        prob_i = 0.0
        if full is True:
            path_iterator = get_all_paths_for_obs(obs_i, path_max_len, rl.env, policy)
        else:
            path_iterator = sim_paths
        for path in path_iterator:
            p_obs = prob_obs(obs_i, path)
            if full is not True and p_obs == 0:
                continue
            if full is True:
                assert p_obs > 0, "Paths should all have positive observation probability, but p({})={}"\
                        .format(path, p_obs)
            p_path = prob_path(path, policy, path_max_len, goal_state, transition_prob)
            if p_path > 0:
                prob_i += p_obs * p_path
            n_paths += 1
        if full is not True:
            if n_paths == 0:
                prob_i = 1.0 / len(sim_paths)  # TODO: what would be correct here?
            else:
                prob_i /= n_paths  # normalize
        assert 0.0 - 1e-10 < prob_i < 1.0 + 1e-10 , "Probability should be between 0 and 1 but was {}"\
                .format(prob_i)
        logprob = np.log(prob_i)
        precomp_obs_logprobs[obs_i] = logprob
        ind_log_obs_probs.append(logprob)
        end_time2 = time.time()
        duration2 = end_time2 - start_time2
        if duration2 > 30.0:
            logger.info("Evaluated loglikelihood of {} in {:.2f}s".format(obs_i, duration2))
        #logger.info("Processed {} paths in {} seconds ({} s/path)"
        #        .format(n_paths, duration2, duration2/n_paths))
    end_time1 = time.time()
    duration1 = end_time1 - start_time1
    #logger.info("Logl evaluated in {} seconds".format(duration1))
    return sum(ind_log_obs_probs) / scale

def get_all_paths_for_obs(obs, path_max_len, env, policy=None):
    """ Returns a tree containing all possible paths that could have generated
        observation 'obs'.
    """
    paths = dict()
    start_time = time.time()
    fill_path_tree(obs, obs.path_len, paths, path_max_len, env, policy)
    end_time = time.time()
    #logger.info("Constructing path tree of depth {} took {} seconds"
    #        .format(obs.path_len, end_time-start_time))
    return PathTreeIterator(obs, paths, obs.path_len)

def prob_path(path, policy, path_max_len, goal_state, transition_prob):
    """ Returns the probability that 'path' would have been generated given 'policy'.

    Parameters
    ----------
    path : list of location tuples [(x0, y0), ..., (xn, yn)]
    policy : callable(state, action) -> p(action | state)
    """
    logp = 0
    if len(path) < path_max_len:
        assert path.transitions[-1].next_state == goal_state  # should have been be pruned
    # assume all start states equally probable
    for transition in path.transitions:
        state = transition.prev_state
        action = transition.action
        next_state = transition.next_state
        act_i_prob = policy(state, action)
        tra_i_prob = transition_prob(transition)
        assert state != goal_state, (state, goal_state)  # should have been pruned
        assert act_i_prob != 0, (state, action)  # should have been pruned
        assert tra_i_prob != 0, (transition)  # should have been pruned
        logp += np.log(act_i_prob) + np.log(tra_i_prob)
    return np.exp(logp)

def prob_obs(obs, path):
    """ Returns the probability that 'path' would generate 'obs'.

    Parameters
    ----------
    obs : tuple (path x0, path y0, path length)
    path : list of location tuples [(x0, y0), ..., (xn, yn)]
    """
    # deterministic summary
    if Observation(path) == obs:
        return 1.0
    return 0.0

def fill_path_tree(obs, full_path_len, paths, path_max_len, env, policy=None):
    """ Recursively fill path tree starting from obs

    Will prune paths that are not feasible:
     * action not possible according to policy
     * goes through the goal state and not end state
     * full path length is less than max, but no way to reach goal state
       with length that is left in obs
    """
    if obs not in paths.keys():
        if obs.path_len > 0:
            node = list()
            for transition in env.get_transitions(obs.start_state):
                if policy is not None and policy(transition.prev_state, transition.action) == 0:
                    # impossible action
                    continue
                next_obs = Observation(start_state=transition.next_state,
                                       path_len=obs.path_len-1)
                if next_obs.path_len > 0 and next_obs.start_state == env.goal_state:
                    # would go through goal state but path does not end there
                    continue
                if full_path_len < path_max_len:
                    # if path is full length we do not know if we reached goal state at the end
                    distance = abs(next_obs.start_state.x - env.goal_state.x) \
                             + abs(next_obs.start_state.y - env.goal_state.y)
                    if next_obs.path_len < distance:
                        # impossible to reach goal state with path of this length
                        continue
                node.append((transition, next_obs))
            if len(node) == 0:
                # dead end
                paths[obs] = ((None, None),)
            else:
                paths[obs] = node
                for transition, next_obs in node:
                    fill_path_tree(next_obs, full_path_len, paths, path_max_len, env, policy)
        else:
            paths[obs] = tuple()

def discrepancy_function(initial_state_generator, *sim_obs, observed=None):
    features = [avg_path_len_by_start(initial_state_generator, i, observed[0])
                for i in range(initial_state_generator.n_initial_states)]
    features_sim = [avg_path_len_by_start(initial_state_generator, i, sim_obs[0])
                    for i in range(initial_state_generator.n_initial_states)]
    disc = 0.0
    for f, fs in zip(features, features_sim):
        disc += np.abs(f - fs)
    disc /= len(features)  # scaling
    return np.array([disc])

def avg_path_len_by_start(initial_state_generator, start_id, obs):
    state = initial_state_generator.get_initial_state(start_id)
    vals = []
    for o in obs:
        if o.start_state == state:
            vals.append(o.path_len)
    if len(vals) > 0:
        return np.mean(vals)
    return 0.0

