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
                max_retries=1):
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
    command_template = "cd {0}; clisp pyramids.lisp {1} {2} {3} {4}"
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
        #logger.warning("Simulation error: {}".format(output))
        raise RuntimeError
    return ret


class LearningModel():
    def __init__(self, params):
        self.p = params

    def __call__(self, *params, random_state=None, index_in_batch=None):
        par = [float(p) for p in params]
        if par[0] > -2.5 or par[0] < -5.0:
            print("RF was {}, clipping to allowed area".format(par[0]))
            par[0] = min(max(-5.0, par[0]), -2.5)
        if par[1] > 0.1 or par[1] < 0.001:
            print("LF was {}, clipping to allowed area".format(par[1]))
            par[1] = min(max(0.001, par[1]), 0.1)
        print("SIM AT {}".format(par))
        for j in range(self.p.max_retries):
            try:
                return _sim(*par, random_state=random_state, index_in_batch=index_in_batch)
            except RuntimeError:
                if j == self.p.max_retries - 1:
                    logger.critical("LISP code crashed, max tries reached, aborting.")
                    assert False
                par = [float(p) + float(random_state.normal(0, 0.01)) for p in params]
                logger.warning("LISP code crashed at {}, retrying nearby at {} ({}/{})".format(params, par, j+1, self.p.max_retries))


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


def get_dataset():   # TODO: test set?
    """ Tenison et al. 2016 Fig. 8, personal communication """
    ret = [
        Summary(1, 3, "encode", 3.0797642498947, 0.304825482333887),
        Summary(1, 3, "solve", 1.21439129477383, 0.360671175339465),
        Summary(1, 3, "respond", 2.74175664346458, 0.444174541617887),
        Summary(1, 4, "encode", 3.41747109018993, 0.358027041765201),
        Summary(1, 4, "solve", 3.63863364012346, 0.853136786547213),
        Summary(1, 4, "respond", 2.72455803940943, 0.392265878871146),
        Summary(1, 5, "encode", 3.27276670381467, 0.3118402309948),
        Summary(1, 5, "solve", 6.18729894954183, 0.614857110436995),
        Summary(1, 5, "respond", 2.58668538361632, 0.176501461114284),
        Summary(2, 3, "encode", 2.26662122373283, 0.321372221955684),
        Summary(2, 3, "solve", 0.0393282673161665, 0.0513864899250276),
        Summary(2, 3, "respond", 2.17913628066846, 0.691433060067407),
        Summary(2, 4, "encode", 2.55794222263096, 0.367054632758945),
        Summary(2, 4, "solve", 0.0572279421746246, 0.053655457800129),
        Summary(2, 4, "respond", 2.44439735956446, 0.298568658991272),
        Summary(2, 5, "encode", 2.50291536061544, 0.562206716261764),
        Summary(2, 5, "solve", 0.100345240286775, 0.109108762746457),
        Summary(2, 5, "respond", 2.32417023349651, 0.229329354269061),
        Summary(3, 3, "encode", 0.576926826579287, 0.421467907940065),
        Summary(3, 3, "solve", 0.0218644206375898, 0.0394008857357585),
        Summary(3, 3, "respond", 2.15572485352515, 0.288231471393689),
        Summary(3, 4, "encode", 0.509769423893816, 0.201090189437526),
        Summary(3, 4, "solve", 0.038717676624672, 0.0551258810188626),
        Summary(3, 4, "respond", 2.30936751274668, 0.161490033598321),
        Summary(3, 5, "encode", 0.530933245232202, 0.246281695856061),
        Summary(3, 5, "solve", 0.0313563718059507, 0.0232321932556025),
        Summary(3, 5, "respond", 2.14674922395738, 0.153517228627859)
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
                if len(vals) > 1:
                    std = float(np.std(vals))
                else:
                    std = 0.0
                ret.append(Summary(stage, height, response, mean, std))
    return ret


def discrepancy_function(*simulated, observed=None):
    disc = 0.0
    n = 0.0
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
                disc += np.abs(osim.mean - oobs.mean) ** 2  # RMSE
                n += 1.0
    disc = np.sqrt(disc / n)  # RMSE
    return np.array([disc])

