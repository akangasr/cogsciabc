import math
import numpy as np
from enum import IntEnum

from sdirl.rl.pybrain_extensions import ParametricLoggingEpisodicTask, ParametricLoggingEnvironment

import logging
logger = logging.getLogger(__name__)


"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Definition of the MDP.
"""

class State():
    """ State of MDP observed by the agent

    Parameters
    ----------
    obs_items : list of MenuItems
    focus : Focus
    click : Click
    quit : Quit
    """
    def __init__(self, obs_items, focus, click, quit):
        self.obs_items = obs_items
        self.focus = focus
        self.click = click
        self.quit = quit

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (tuple(self.obs_items) + (self.focus, self.click, self.quit)).__hash__()

    def __repr__(self):
        return "({},{},{},{})".format(self.obs_items, self.focus, self.click, self.quit)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return State([item.copy() for item in self.obs_items], self.focus, self.click, self.quit)

class ItemRelevance(IntEnum):
    NOT_OBSERVED = 0
    TARGET_RELEVANCE = 1  # 1.0
    HIGH_RELEVANCE = 2  # 0.6
    MED_RELEVANCE = 3  # 0.3
    LOW_RELEVANCE = 4  # 0.0

class ItemLength(IntEnum):
    NOT_OBSERVED = 0
    TARGET_LENGTH = 1
    NOT_TARGET_LENGTH = 2

class MenuItem():
    """ Single menu item

    Parameters
    ----------
    item_relevance : ItemRelevance
    item_length : ItemLength
    """
    def __init__(self, item_relevance, item_length):
        self.item_relevance = item_relevance
        self.item_length = item_length

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (int(self.item_relevance), int(self.item_length)).__hash__()

    def __repr__(self):
        return "({},{})".format(self.item_relevance, self.item_length)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return MenuItem(self.item_relevance, self.item_length)

class Quit(IntEnum):
    NOT_QUIT = 0
    HAS_QUIT = 1

class Focus(IntEnum):  # assume 8 items in menu
    ITEM_1 = 0
    ITEM_2 = 1
    ITEM_3 = 2
    ITEM_4 = 3
    ITEM_5 = 4
    ITEM_6 = 5
    ITEM_7 = 6
    ITEM_8 = 7
    ABOVE_MENU = 8

class Click(IntEnum):  # assume 8 items in menu
    CLICK_1 = 0
    CLICK_2 = 1
    CLICK_3 = 2
    CLICK_4 = 3
    CLICK_5 = 4
    CLICK_6 = 5
    CLICK_7 = 6
    CLICK_8 = 7
    NOT_CLICKED = 8

class Action(IntEnum):  # assume 8 items in menu
    LOOK_1 = 0
    LOOK_2 = 1
    LOOK_3 = 2
    LOOK_4 = 3
    LOOK_5 = 4
    LOOK_6 = 5
    LOOK_7 = 6
    LOOK_8 = 7
    CLICK = 8
    QUIT = 9


class SearchTask(ParametricLoggingEpisodicTask):

    reward_success = 10000
    reward_failure = -10000

    def __init__(self, env, max_number_of_actions_per_session):
        super(SearchTask, self).__init__(env)

        self.reward_success = 10000
        self.reward_failure = -10000
        self.max_number_of_actions_per_session = max_number_of_actions_per_session

    def to_dict(self):
        return {
                "max_number_of_actions_per_session": self.max_number_of_actions_per_session
                }

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        if self.env.state.click != Click.NOT_CLICKED:
            if self.env.clicked_item.item_relevance == ItemRelevance.TARGET_RELEVANCE:
                # reward for clicking the correct item after seeing it
                # target item should always have correct length
                return self.reward_success
            else:
                # penalty for clicking the wrong item
                return self.reward_failure
        elif self.env.state.quit == Quit.HAS_QUIT:
            if self.env.target_present is False:
                # reward for quitting when target is absent
                return self.reward_success
            else:
                # penalty for quitting when target is present
                return self.reward_failure
        # default penalty for spending time
        return int(-1 * self.env.action_duration)

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.state.click != Click.NOT_CLICKED:
            # click ends task
            return True
        elif self.env.state.quit == Quit.HAS_QUIT:
            # quit ends task
            return True
        return False


class SearchEnvironment(ParametricLoggingEnvironment):

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
            n_training_menus=10000):
        """ Initializes the search environment
        """
        self.v = None # set with setup
        self.random_state = None # set with setup
        self.log = None # set by RL_model
        self.task = None # set by Task

        self.menu_type = menu_type
        self.menu_groups = menu_groups
        self.menu_items_per_group = menu_items_per_group
        self.n_items = self.menu_groups * self.menu_items_per_group
        assert self.n_items == 8
        self.semantic_levels = semantic_levels
        self.gap_between_items = gap_between_items
        self.prop_target_absent = prop_target_absent
        self.length_observations = length_observations
        self.p_obs_len_cur = p_obs_len_cur
        self.p_obs_len_adj = p_obs_len_adj
        self.n_training_menus = n_training_menus
        self.training_menus = list()
        self.training = True
        self.n_item_lengths = 3
        self.log_session_variables = ["items", "target_present", "target_idx"]
        self.log_step_variables = ["duration_focus_ms",
                                   "duration_saccade_ms",
                                   "action_duration",
                                   "action",
                                   "gaze_location"]

        # technical variables
        self.discreteStates = True
        self.outdim = 1
        self.indim = 1
        self.discreteActions = True
        self.numActions = self.n_items + 2 # look + click + quit

    def to_dict(self):
        return {
                "menu_type": self.menu_type,
                "menu_groups": self.menu_groups,
                "menu_items_per_group": self.menu_items_per_group,
                "semantic_levels": self.semantic_levels,
                "gap_between_items": self.gap_between_items,
                "prop_target_absent": self.prop_target_absent,
                "length_observations": self.length_observations,
                "n_training_menus": self.n_training_menus,
                }

    def _get_menu(self):
        if self.training is True and len(self.training_menus) >= self.n_training_menus:
            idx = self.random_state.randint(self.n_training_menus)
            return self.training_menus[idx]
        # generate menu item semantic relevances and lengths
        items = list()
        if self.menu_type == "semantic":
            items, target_idx = self._get_semantic_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.semantic_levels,
                        self.prop_target_absent)
        elif self.menu_type == "unordered":
            items, target_idx = self._get_unordered_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.semantic_levels,
                        self.prop_target_absent)
        else:
            raise ValueError("Unknown menu type: {}".format(self.menu_type))
        lengths = self.random_state.randint(0,self.n_item_lengths,len(items)).tolist()
        target_present = (target_idx != None)
        if target_present:
            items[target_idx].item_relevance = ItemRelevance.TARGET_RELEVANCE
            target_len = lengths[target_idx]
        else:
            # target not present, choose target length randomly
            target_len = self.random_state.randint(0,self.n_item_lengths)
        for i, length in enumerate(lengths):
            if length == target_len:
                items[i].item_length = ItemLength.TARGET_LENGTH
            else:
                items[i].item_length = ItemLength.NOT_TARGET_LENGTH
        menu = (tuple(items), target_present, target_idx)
        if self.training is True:
            self.training_menus.append(menu)
        return menu

    def reset(self):
        """ Called by the library to reset the state
        """
        # state hidden from agent
        self.items, self.target_present, self.target_idx = self._get_menu()

        # state observed by agent
        obs_items = [MenuItem(ItemRelevance.NOT_OBSERVED, ItemLength.NOT_OBSERVED) for i in range(self.n_items)]
        focus = Focus.ABOVE_MENU
        click = Click.NOT_CLICKED
        quit = Quit.NOT_QUIT
        self.state = State(obs_items, focus, click, quit)
        self.prev_state = self.state.copy()

        # misc environment state variables
        self.action_duration = None
        self.duration_focus_ms = None
        self.duration_saccade_ms = None
        self.action = None
        self.gaze_location = None
        self.n_actions = 0
        self.item_locations = np.arange(self.gap_between_items, self.gap_between_items*(self.n_items+2), self.gap_between_items)
        self._start_log_for_new_session()

    def performAction(self, action):
        """ Changes the state of the environment based on agent action """
        self.action = Action(int(action[0]))
        self.prev_state = self.state.copy()
        self.state, self.duration_focus_ms, self.duration_saccade_ms = self.do_transition(self.state, self.action)
        self.action_duration = self.duration_focus_ms + self.duration_saccade_ms
        self.gaze_location = int(self.state.focus)
        self.n_actions += 1
        self._log_transition()

    def _observe_relevance_at(self, state, focus):
        state.obs_items[focus].item_relevance = self.items[focus].item_relevance
        return state

    def _observe_length_at(self, state, focus):
        state.obs_items[focus].item_length = self.items[focus].item_length
        return state

    def do_transition(self, state, action):
        """ Changes the state of the environment based on agent action.
            Also depends on the unobserved state of the environment.

        Parameters
        ----------
        state : State
        action : Action

        Returns
        -------
        tuple (State, int) with new state and action duration in ms
        """
        state = state.copy()
        # menu recall event may happen at first action
        if self.n_actions == 0:
            if "menu_recall_probability" in self.v and self.random_state.rand() < float(self.v["menu_recall_probability"]):
                state.obs_items = [item.copy() for item in self.items]

        # observe items's state
        if action != Action.CLICK and action != Action.QUIT:
            # saccade
            # item_locations are off-by-one to other lists
            if state.focus != Focus.ABOVE_MENU:
                amplitude = abs(self.item_locations[int(state.focus)+1] - self.item_locations[int(action)+1])
            else:
                amplitude = abs(self.item_locations[0] - self.item_locations[int(action)+1])
            saccade_duration = int(37 + 2.7 * amplitude)
            state.focus = Focus(int(action))  # assume these match

            # fixation
            if "focus_duration_100ms" in self.v:
                focus_duration = int(self.v["focus_duration_100ms"] * 100)
            else:
                focus_duration = 400
            # semantic observation at focus
            state = self._observe_relevance_at(state, int(state.focus))
            # possible length observations
            if self.length_observations is True:
                if int(state.focus) > 0 and self.random_state.rand() < self.p_obs_len_adj:
                    state = self._observe_length_at(state, int(state.focus)-1)
                if self.random_state.rand() < self.p_obs_len_cur:
                    state = self._observe_length_at(state, int(state.focus))
                if int(state.focus) < self.n_items-1 and self.random_state.rand() < self.p_obs_len_adj:
                    state = self._observe_length_at(state, int(state.focus)+1)
            # possible semantic peripheral observations
            if "prob_obs_adjacent" in self.v:
                if int(state.focus) > 0 and self.random_state.rand() < float(self.v["prob_obs_adjacent"]):
                    state = self._observe_relevance_at(state, int(state.focus)-1)
                if int(state.focus) < self.n_items-1 and self.random_state.rand() < float(self.v["prob_obs_adjacent"]):
                    state = self._observe_relevance_at(state, int(state.focus)+1)

        # choose item
        elif action == Action.CLICK:
            if state.focus != Focus.ABOVE_MENU:
                state.click = Click(int(state.focus))  # assume these match
            else:
                # trying to select an item when not focusing on any item equals to quitting
                state.quit = Quit.HAS_QUIT
            if "selection_delay_s" in self.v:
                focus_duration = int(self.v["selection_delay_s"] * 1000)
            else:
                focus_duration = 0
            saccade_duration = 0

        # quit without choosing any item
        elif action == Action.QUIT:
            state.quit = Quit.HAS_QUIT
            focus_duration = 0
            saccade_duration = 0

        else:
            raise ValueError("Unknown action: {}".format(action))

        return state, focus_duration, saccade_duration

    @property
    def clicked_item(self):
        if self.state.click == Click.NOT_CLICKED:
            return None
        return self.items[int(self.state.click)]  # assume indexes aligned

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return [self.state.__hash__()]  # needs to return a list

    def _semantic(self, n_groups, n_each_group, p_absent):
        n_items = n_groups * n_each_group
        target_value = 1

        """alpha and beta parameters for the menus with no target"""
        absent_menu_parameters = [2.1422, 13.4426]

        """alpha and beta for non-target/irrelevant menu items"""
        non_target_group_paremeters = [5.3665, 18.8826]

        """alpha and beta for target/relevant menu items"""
        target_group_parameters = [3.1625, 1.2766]

        semantic_menu = np.array([0] * n_items)[np.newaxis]

        """randomly select whether the target is present or abscent"""
        target_type = self.random_state.rand()
        target_location = self.random_state.randint(0, n_items)

        if target_type > p_absent:
            target_group_samples = self.random_state.beta \
                (target_group_parameters[0], target_group_parameters[1], (1, n_each_group))[0]
            """sample distractors from the Distractor group distribution"""
            distractor_group_samples = self.random_state.beta \
                (non_target_group_paremeters[0], non_target_group_paremeters[1], (1, n_items))[0];

            """ step 3 using the samples above to create Organised Menu and Random Menu
                and then add the target group
                the menu is created with all distractors first
            """
            menu1 = distractor_group_samples
            target_in_group = math.ceil((target_location + 1) / float(n_each_group))
            begin = (target_in_group - 1) * n_each_group
            end = (target_in_group - 1) * n_each_group + n_each_group

            menu1[begin:end] = target_group_samples
            menu1[target_location] = target_value
        else:
            target_location = None
            menu1 = self.random_state.beta\
                (absent_menu_parameters[0],\
                 absent_menu_parameters[1],\
                 (1, n_items))

        semantic_menu = menu1
        return semantic_menu, target_location

    def _get_unordered_menu(self, n_groups, n_each_group, n_grids, p_absent):
        assert(n_groups > 1)
        assert(n_each_group > 1)
        assert(n_grids > 0)
        semantic_menu, target = self._semantic(n_groups, n_each_group, p_absent)
        unordered_menu = self.random_state.permutation(semantic_menu)
        gridded_menu = self._griding(unordered_menu, target, n_grids)
        menu_length = n_each_group * n_groups
        coded_menu = [MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED) for i in range(menu_length)]
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[1]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.MED_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[2]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.HIGH_RELEVANCE, ItemLength.NOT_OBSERVED)
                count += 1
        return coded_menu, target

    def _griding(self, menu, target, n_levels):
        start = 1 / float(2 * n_levels)
        stop = 1
        step = 1 / float(n_levels)
        np_menu = np.array(menu)[np.newaxis]
        griding_semantic_levels = np.arange(start, stop, step)
        temp_levels = abs(griding_semantic_levels - np_menu.T)
        if target != None:
            min_index = temp_levels.argmin(axis=1)
            gridded_menu = griding_semantic_levels[min_index]
            gridded_menu[target] = 1
        else:
            min_index = temp_levels.argmin(axis=2)
            gridded_menu = griding_semantic_levels[min_index]
        return gridded_menu.T

    def _get_semantic_menu(self, n_groups, n_each_group, n_grids, p_absent):
        assert(n_groups > 0)
        assert(n_each_group > 0)
        assert(n_grids > 0)
        menu, target = self._semantic(n_groups, n_each_group, p_absent)
        gridded_menu = self._griding(menu, target, n_grids)
        menu_length = n_each_group*n_groups
        coded_menu = [MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED) for i in range(menu_length)]
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[1]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.MED_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[2]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.HIGH_RELEVANCE, ItemLength.NOT_OBSERVED)
                count += 1
        return coded_menu, target

