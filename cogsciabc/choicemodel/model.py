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

""" DFT choice model
    Busemeyer & Townsend 1993
"""

class ChoiceParams():

   def __init__(self,
                n_trajectories=1000,
                timestep=0.01,
                t_max=3,
                A_gain=6000,
                A_loss=0,
                B_gain=3000,
                B_loss=0,
                A_prob=0.45,
                B_prob=1.0):
        for k, v in locals().items():
            setattr(self, k, v)


class Observation(Serializable):
    def __init__(self, choice, duration):
        super().__init__(self)
        self.choice = choice
        self.duration = float(duration)

    def __repr__(self):
        return "{}".format(self)

    def __str__(self):
        return "Obs({} {})".format(self.choice, self.duration)


class Summary(Serializable):
    def __init__(self, prob_A, prob_to, mean_A, std_A, mean_B, std_B):
        super().__init__(self)
        self.prob_A = float(prob_A)
        self.prob_timeout = float(prob_to)
        self.mean_A = float(mean_A)
        self.std_A = float(std_A)
        self.mean_B = float(mean_B)
        self.std_B = float(std_B)

    def __repr__(self):
        return "{}".format(self)

    def __str__(self):
        return "Sum(p{} A{:.2f}+-{:.2f} p{} B{:.2f}+-{:.2f} p{} TO)".format(
                self.prob_A, self.mean_A, self.std_A, 1-self.prob_A-self.prob_timeout, self.mean_B, self.std_B, self.prob_timeout)


class ChoiceModel():
    def __init__(self, params):
        self.p = params

    def __call__(self, *params, random_state=None, index_in_batch=None):
        par = [float(p) for p in params]
        s = par[0]  # growth-decay rate
        theta = par[1]  # selection threshold
        a = par[2] # gain gradient
        b = par[3] # loss gradient
        alpha = par[4] # value deprecation
        print("SIM AT s={} theta={} a={} b={} alpha={}".format(*par))
        t = 0.0
        h = float(self.p.timestep)
        z = 0.0 # initial location of random walk
        P = z  # TODO: draw?
        # subjective values and probabilities
        u_A_gain = self.u(self.p.A_gain, alpha)
        u_A_loss = self.u(self.p.A_loss, alpha)
        u_B_gain = self.u(self.p.B_gain, alpha)
        u_B_loss = self.u(self.p.B_loss, alpha)
        p_A_gain = self.p.A_prob
        p_A_loss = 1.0 - self.p.A_prob
        p_B_gain = self.p.B_prob
        p_B_loss = 1.0 - self.p.B_prob
        # expected utilities
        avg_gain_A = p_A_gain * u_A_gain
        avg_loss_A = p_A_loss * u_A_loss
        avg_gain_B = p_B_gain * u_B_gain
        avg_loss_B = p_B_loss * u_B_loss
        avg_gains = avg_gain_A + avg_gain_B
        avg_losses = avg_loss_A + avg_loss_B
        avg_utility_A = avg_gain_A + avg_loss_A
        avg_utility_B = avg_gain_B + avg_loss_B
        diff_u_A_gain = u_A_gain - avg_utility_A
        diff_u_A_loss = u_A_loss - avg_utility_A
        diff_u_B_gain = u_B_gain - avg_utility_B
        diff_u_B_loss = u_B_loss - avg_utility_B
        diff_gain_AB = avg_gain_A - avg_gain_B
        diff_loss_AB = avg_loss_A - avg_loss_B
        # variances
        sigma2_A = p_A_gain * diff_u_A_gain**2 + \
                   p_A_loss * diff_u_A_loss**2
        sigma2_B = p_B_gain * diff_u_B_gain**2 + \
                   p_B_loss * diff_u_B_loss**2
        covar_AB = p_A_gain * p_B_gain * diff_u_A_gain * diff_u_B_gain + \
                   p_A_gain * p_B_loss * diff_u_A_gain * diff_u_B_loss + \
                   p_A_loss * p_B_gain * diff_u_A_loss * diff_u_B_gain + \
                   p_A_loss * p_B_loss * diff_u_A_loss * diff_u_B_loss
        sigma2 = sigma2_A + sigma2_B - 2*covar_AB  # variance of valence difference
        # goal gradient
        c = b * avg_losses - a * avg_gains
        # mean valence input
        delta = diff_gain_AB * (1 - a * theta) + diff_loss_AB * (1 - b * theta)
        # decay
        decay = 1 - (s + c) * h
        # step
        step = delta * h

        obs = list()
        for i in range(self.p.n_trajectories):
            o = Observation("X", self.p.t_max)
            while t < self.p.t_max:
                # residual input
                epsilon = float(random_state.normal(0, np.sqrt(h * sigma2)))
                # preference
                P_last = P
                t += h
                P = decay * P_last + step + epsilon
                if P > theta:
                    o = Observation("A", t)
                    break
                if P < -theta:
                    o = Observation("B", t)
                    break
            obs.append(o)
            t = 0.0
            P = z  # TODO: draw?
        return obs

    def u(self, val, alpha):
        if val > 0:
            return val ** alpha
        return -2 * abs(val)**alpha


def get_model(p, elfi_p, observation):
    model = elfi_p[0].model
    cm = ChoiceModel(p)
    simulator = elfi.Simulator(elfi.tools.vectorize(cm),
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
    print(data)


def get_dataset():
    """ Test data """
    return [Summary(0.14, 0.0, 1.2, 0.8, 1.2, 0.8)]


def summary_function(obs):
    counts = {"A": 0, "B": 0, "X": 0}
    durations = {"A": list(), "B": list(), "X": list()}
    for o in obs:
        counts[o.choice] += 1
        durations[o.choice].append(o.duration)
    pA = counts["A"] / float(len(obs))
    pTO = counts["X"] / float(len(obs))
    if len(durations["A"]) == 0:
        meanA = -1.0
        stdA = -1.0
    else:
        meanA = np.mean(durations["A"])
        stdA = np.std(durations["A"])
    if len(durations["B"]) == 0:
        meanB = -1.0
        stdB = -1.0
    else:
        meanB = np.mean(durations["B"])
        stdB = np.std(durations["B"])
    return [Summary(pA, pTO, meanA, stdA, meanB, stdB)]


def discrepancy_function(*simulated, observed=None):
    s = simulated[0][0]
    o = observed[0][0]
    disc = np.abs(o.prob_A - s.prob_A) ** 2 + \
           np.abs(o.prob_timeout - s.prob_timeout) ** 2 + \
           np.abs(o.mean_A - s.mean_A) ** 2 + \
           np.abs(o.std_A - s.std_A) ** 2 + \
           np.abs(o.mean_B - s.mean_B) ** 2 + \
           np.abs(o.std_B - s.std_B) ** 2
    disc = np.sqrt(disc / 6.0)  # RMSE
    if disc != disc:
        print("SIM", s)
        print("OBS", o)
        disc = 1e10
    return np.array([disc])

