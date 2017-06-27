import os
import traceback
import subprocess
import numpy as np

import elfi
from elfie.serializable import Serializable

import matplotlib
from matplotlib import pyplot as pl

import logging
logger = logging.getLogger(__name__)

""" Pyramid task learning ACT-R model
    Tenison et al. 2016 http://act-r.psy.cmu.edu/?post_type=publications&p=19047
"""

class LearningParams():

   def __init__(self,
                sample_size=10,
                sample_d=0.05,
                max_retries=1,
                bounds=dict()):
        for k, v in locals().items():
            setattr(self, k, v)


class Observation(Serializable):
    def __init__(self, stage, height, response, val):
        super().__init__(self)
        self.stage = int(stage)
        self.height = int(height)
        self.response = "{}".format(response).strip("\"")
        self.val = float(val)

    def __repr__(self):
        return "{}".format(self)

    def __str__(self):
        return "Obs(s{} h{} {} = {:.2f})".format(
                self.stage, self.height, self.response, self.val)


class Summary(Serializable):
    def __init__(self, stage, height, response, mean, std):
        super().__init__(self)
        self.stage = int(stage)
        self.height = int(height)
        self.response = "{}".format(response).strip("\"")
        self.mean = float(mean)
        self.std = float(std)

    def __repr__(self):
        return "{}".format(self)

    def __str__(self):
        return "Sum(s{} h{} {} = {:.2f} +- {:.2f})".format(
                self.stage, self.height, self.response, self.mean, self.std)


def _sim(*params, random_state=None, index_in_batch=None):
    cmd = _create_command(*params)
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)
    output = process.communicate()[0].decode("utf-8")
    return _parse_results(output)


def _create_command(*params, actr_dir="actr6"):
    command_template = "cd {0}; clisp pyramids.lisp {1} {2}"
    source_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(source_dir, actr_dir)
    return command_template.format(model_dir, *params)


def _parse_results(output, result_tag="RESULT_OUT", tag_separator=" ", item_separator=","):
    lines = output.splitlines()
    features = []
    ret = []
    for line in lines:
        tags = line.split(tag_separator, 2)
        if tags[0] == result_tag:
            items = tags[1].split(item_separator)
            if len(items) != 4:
                logger.critical("Assumed to receive 4 features and one response value, got: {}".format(line))
                assert False
            if len(features) == 0:
                # assume first line with caption
                features = items
                continue
            kwargs = {}
            for i in range(4):
                kwargs[features[i]] = items[i]
            ret.append(Observation(**kwargs))
    if len(ret) == 0:
        logger.warning("Simulation error: {}".format(output))
        raise RuntimeError
    return ret


class LearningModel():
    def __init__(self, params):
        self.p = params

    def __call__(self, *params, random_state=None, index_in_batch=None):
        ret = []
        bounds = [self.p.bounds[k] for k in sorted(self.p.bounds.keys())]
        for i in range(self.p.sample_size):
            for i in range(self.p.max_retries):
                try:
                    new_params = [min(b[1], max(b[0], p + random_state.uniform(-self.p.sample_d, self.p.sample_d))) for p, b in zip(params, bounds)]
                    ret += _sim(*new_params, random_state=random_state, index_in_batch=index_in_batch)
                    break
                except RuntimeError:
                    if i == self.p.max_retries - 1:
                        logger.critical("LISP code crashed, max tries reached, aborting.")
                        assert False
                    logger.warning("LISP code crashed near {}, retrying ({}/{})".format(params, i, self.p.max_retries))
        return ret


def get_model(p, elfi_p, observation):
    model = elfi_p[0].model
    lm = LearningModel(p)
    simulator = elfi.Simulator(elfi.tools.vectorize(lm),
                               *elfi_p,
                               model=model,
                               name="simulator")
    summary = elfi.Summary(elfi.tools.vectorize(summary_function),
                           simulator,
                           model=model,
                           observed=observation,
                           name="summary")
    discrepancy = elfi.Discrepancy(elfi.tools.vectorize(discrepancy_function),
                                   summary,
                                   model=model,
                                   name="discrepancy")
    return model


def plot_data(pdf, figsize, data, suptitle):
    fs = (figsize[0], figsize[0]/2.0)
    fig, axarr = pl.subplots(1,3,figsize=fs)
    fig.suptitle(suptitle)
    try:
        i = 0
        for stage in [1,2,3]:
            responses = ["encode", "solve", "respond"]
            ax = axarr[i]
            ax.set_title("Learning Phase {}".format(stage))
            ax.set_ylim(-0.5, 10)
            ax.set_xlim(-0.5, 2.5)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(responses)
            if i == 0:
                ax.set_ylabel("Mean stage duration (Sec.)")
            if i == 1:
                ax.set_xlabel("Cognitive Stages within Learning Phases")
            for height, color in zip([3,4,5], ["orange", "green", "blue"]):
                x = [0, 1, 2]
                means = list()
                stds = list()
                for response in responses:
                    for s in data:
                        if s.stage == stage and s.height == height and s.response == response:
                            means.append(s.mean)
                            stds.append(s.std)
                ax.errorbar(x, means, yerr=stds, fmt='-o', color=color, label="Height {}".format(height))
            i += 1
        pl.tight_layout(pad=2.0)
    except Exception as e:
        fig.text(0.02, 0.02, "Was not able to plot data: {}".format(e))
        tb = traceback.format_exc()
        logger.critical(tb)
    pdf.savefig()
    pl.close()


def get_dataset():
    """ Tenison et al. 2016 Fig. 8, estimated """
    ret = [
        Summary(1, 3, "encode", 3.26, 0.5),
        Summary(1, 3, "solve", 1.3, 0.5),
        Summary(1, 3, "respond", 2.68, 0.5),
        Summary(1, 4, "encode", 3.26, 0.5),
        Summary(1, 4, "solve", 3.8, 1.0),
        Summary(1, 4, "respond", 2.68, 0.5),
        Summary(1, 5, "encode", 3.26, 0.5),
        Summary(1, 5, "solve", 6.2, 0.7),
        Summary(1, 5, "respond", 2.68, 0.5),
        Summary(2, 3, "encode", 2.44, 0.5),
        Summary(2, 3, "solve", 0.07, 0.02),
        Summary(2, 3, "respond", 2.32, 0.5),
        Summary(2, 4, "encode", 2.44, 0.5),
        Summary(2, 4, "solve", 0.07, 0.02),
        Summary(2, 4, "respond", 2.32, 0.5),
        Summary(2, 5, "encode", 2.44, 0.5),
        Summary(2, 5, "solve", 0.07, 0.02),
        Summary(2, 5, "respond", 2.32, 0.5),
        Summary(3, 3, "encode", 0.54, 0.3),
        Summary(3, 3, "solve", 0.03, 0.01),
        Summary(3, 3, "respond", 2.2, 0.2),
        Summary(3, 4, "encode", 0.54, 0.3),
        Summary(3, 4, "solve", 0.03, 0.01),
        Summary(3, 4, "respond", 2.2, 0.2),
        Summary(3, 5, "encode", 0.54, 0.3),
        Summary(3, 5, "solve", 0.03, 0.01),
        Summary(3, 5, "respond", 2.2, 0.2)
        ]
    return ret


def summary_function(obs):
    ret = []
    for stage in [1,2,3]:
        for height in [3,4,5]:
            for response in ["encode", "solve", "respond"]:
                vals = []
                for o in obs:
                    if o.stage == stage and o.height == height and o.response == response:
                        vals.append(o.val)
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                ret.append(Summary(stage, height, response, mean, std))
    return ret


def discrepancy_function(*simulated, observed=None):
    disc = 0
    for stage in [1,2,3]:
        for height in [3,4,5]:
            for response in ["encode", "solve", "respond"]:
                for o in simulated[0]:
                    if o.stage == stage and o.height == height and o.response == response:
                        osim = o
                        break
                for o in observed[0]:
                    if o.stage == stage and o.height == height and o.response == response:
                        oobs = o
                        break
                d = np.abs(osim.mean - oobs.mean) ** 2 + np.abs(osim.std - oobs.std)
                disc += float(d)
    return np.array([disc])

