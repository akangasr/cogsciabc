import sys
import os
import re
import fnmatch
import numpy as np
import scipy as sp
import random
import copy

import matplotlib
import matplotlib.pyplot as pl

from elfie.analysis import ExperimentLog, ExperimentGroup
from elfie.utils import pretty_time

import warnings
warnings.filterwarnings("error")

import logging
logger = logging.getLogger(__name__)

def collect_experiments_from_directory(directory, script, verbose=False):
    print("Loading experiments from '{}'".format(directory))
    subdirs = [x[0] for x in os.walk(directory)]
    experiment_logs = dict()
    if len(subdirs) == 0:
        print("Path '{}' has no subdirectories".format(directory))
        return -1
    for subdir in subdirs:
        if subdir == directory:
            continue
        m = re.match(r"{}/([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)_(\d+)".format(directory), subdir)
        if m is None:
            if verbose is True:
                print("skip (unknown) .. {}".format(subdir))
        else:
            script_id, method, samples, n = m.groups()
            if fnmatch.fnmatch(script_id, script) is False:
                if verbose is True:
                    print("skip ({}) .. {}".format(script_id, subdir))
                continue
            samples = int(samples)
            path = os.path.join(subdir, "results.json")
            if not os.path.exists(path):
                if verbose is True:
                    print("no results .. {}".format(path))
                continue
            try:
                exp = ExperimentLog(path, method, samples, exclude=["MD_sim", "ML_sim", "MAP_sim", "PM_sim", "LM_sim"])
                if script_id not in experiment_logs.keys():
                    experiment_logs[script_id] = list()
                experiment_logs[script_id].append(exp)
                if verbose is True:
                    print("ok .. {}".format(path))
            except Exception as e:
                if verbose is True:
                    print("error .. {}".format(path))
                print(e)
    print("")
    groups = dict()
    for k in sorted(experiment_logs.keys()):
        v = experiment_logs[k]
        groups[k] = ExperimentGroup(v)
        print("{} - {} experiments loaded".format(k, len(v)))
        for m in sorted(groups[k].methods.keys()):
            print(" - {} - {} experiments".format(m, len(groups[k].methods[m])))
    return groups


def _get_duration(e):
    if e.method in ["lbfgsb", "neldermead", "neldermead (parallel)"]:
        return e.sampling_duration
    if e.method in ["grid", "bo"]:
        return e.sampling_duration * e.n_cores

def cs_relabeler(label):
    if label == "BO MED":
        return "BO"
    if label == "BO ML":
        return "ABC ML"
    if label == "BO LM":
        return "ABC LM"
    if label == "BO PM":
        return "ABC PM"
    if label == "BO MAP":
        return "ABC MAP"
    if label == "BO LIK":
        return "ABC LIK"
    if label == "BO POST":
        return "ABC POST"
    return label

def irl_relabeler1(label):
    if label == "A ML":
        return "ABC"
    if label == "AL ML":
        return "ABCL"
    if label == "S MED":
        return "MC1K"
    if label == "SL MED":
        return "MC50K"
    if label == "E MED":
        return "EXACT"
    if label == "R":
        return "RANDOM"
    return label

def irl_relabeler2(label):
    if label == "g7f2":
        return "7x7"
    if label == "g7f3":
        return "7x7"
    if label == "g9f2":
        return "9x9"
    if label == "g9f3":
        return "9x9"
    if label == "gl9f3":
        return "9x9"
    if label == "g11f2":
        return "11x11"
    if label == "g11f3":
        return "11x11"
    if label == "gl11f3":
        return "11x11"
    if label == "g13f2":
        return "13x13"
    if label == "g13f3":
        return "13x13"
    if label == "g21f2":
        return "21x21"
    if label == "g21f3":
        return "21x21"
    if label == "gl21f3":
        return "21x21"
    if label == "g31f2":
        return "31x31"
    if label == "g31f3":
        return "31x31"
    if label == "gl31f3":
        return "31x31"
    if label == "g51f2":
        return "51x51"
    if label == "g51f3":
        return "51x51"
    return label


def relabel(d, rl):
    ret = dict()
    for k, v in d.items():
        ret[rl(k)] = v
    return ret


class Plotdef():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def plot_trendlines(datas, pd):
    fig = pl.figure(figsize=pd.figsize)
    #pl.rc('text', usetex=True)
    #pl.rc('font', **{'family':'sans-serif','sans-serif':['Avant Garde']})
    for label, vals in datas.items():
        means, stds, x, p05s, p95s, ns = vals
        pl.plot(x, means, marker=pd.markers[label], color=pd.colors[label], label=label)
        if pd.errbars is True:
            pl.fill_between(x, p05s, p95s, facecolor=pd.colors[label], alpha=pd.alpha)
        if hasattr(pd, "ylabel"):
            pl.ylabel(pd.ylabel)
        if hasattr(pd, "xlabel"):
            pl.xlabel(pd.xlabel)
    pl.title(pd.title)
    pl.legend(loc=pd.legend_loc)
    pl.show()

def plot_barchart(datas, pd):
    ind = np.arange(pd.bars+1)
    bar_width = 1.0

    fig, ax = pl.subplots(figsize=pd.figsize)
    #pl.rc('text', usetex=True)
    #pl.rc('font', **{'family':'sans-serif','sans-serif':['Avant Garde']})
    ax.grid(True)
    ax.grid(zorder=0)
    for line in ax.get_xgridlines():
        line.set_color("white")
        line.set_linestyle("")
    for line in ax.get_ygridlines():
        line.set_color("lightgrey")
        line.set_linestyle("-")

    bars = list()
    labels = list()
    i = 1
    j = 0
    for label in pd.order_g:
        if label not in datas.keys():
            #print("Skip {}".format(label))
            continue
        for method in pd.order_m:
            if method not in datas[label].keys():
                #print("Skip {} {}".format(label, method))
                continue
            if label == method:
                labels.append("{}".format(label))
            else:
                labels.append("{} {}".format(label, method))
            means, stds, _, _, _, _ = datas[label][method]
            #print(label, method)
            bar = ax.bar(ind[i] + j*bar_width/2, means[0], bar_width, color=pd.colors[label],
                         hatch=pd.hatches[method], edgecolor="black", zorder=3)
            if hasattr(pd, "errbars"):
                ax.errorbar(ind[i] + j*bar_width/2, means[0], fmt=" ",
                         yerr=stds[0], ecolor="black", capsize=5, zorder=4)
            i += 1
            bars.append(bar)
        j += 1

    ax.set_ylabel(pd.ylabel, fontsize=16)
    ax.set_title(pd.title, fontsize=20)
    ax.set_xticks([])
    #ax.set_xticks(ind[1:])
    #ax.set_xticklabels([k for k in labels], fontsize=16)
    pl.tick_params(axis='x',which='both',bottom='off',top='off')
    pl.setp(ax.get_yticklabels(), fontsize=16)
    pl.xlim(ind[0]-0.1, ind[-1]+0.1 + bar_width + j*bar_width/2)

    if hasattr(pd, "ylim"):
        pl.ylim(pd.ylim)
    if hasattr(pd, "yscale"):
        pl.yscale(pd.yscale)

    if pd.legend_loc == "in":
        ax.legend([b[0] for b in bars], [k for k in labels], loc=2,
              ncol=pd.legend_cols, fontsize=16)
    if pd.legend_loc == "out":
        ax.legend([b[0] for b in bars], [k for k in labels], loc='upper center',
              bbox_to_anchor=(0.5, -0.1), ncol=pd.legend_cols, fontsize=16)
        fig.subplots_adjust(bottom=0.5)
    pl.show()

def do_ttests(datas):
    for label in sorted(datas.keys()):
        print("{}".format(label))
        for k1 in sorted(datas[label].keys()):
            for k2 in sorted(datas[label].keys()):
                if k1 >= k2:
                    continue
                m1, s1, _, _, _, n1 = datas[label][k1]
                m2, s2, _, _, _, n2 = datas[label][k2]
                stat, p = sp.stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
                if p < 0.001:
                    mark = "***"
                elif p < 0.01:
                    mark = "**"
                elif p < 0.05:
                    mark = "*"
                else:
                    mark = ""
                print("- T-test between {} and {}: p={:.4f} {}".format(k1, k2, float(p), mark))


def simulate_parallel(eg, name, fit_value, test_value, duration, n_par):
    print("Treating {} as if it would have been run {} processes in parallel".format(name, n_par))
    allexp = eg.exp
    for n_samples in eg.n_samples:
        exp = [e for e in eg.methods[name] if e.n_samples == n_samples]
        if len(exp) < 1:
            continue
        print("{} experiments where n_samples = {}".format(len(exp), n_samples))
        vals = [(getattr(e, fit_value), getattr(e, test_value)) for e in exp]
        for e in exp:
            e2 = copy.deepcopy(e)
            setattr(e2, "method", "{} (parallel)".format(name))
            d = getattr(e, duration)
            setattr(e2, duration, d*n_par)
            v = random.sample(vals, n_par)
            min_fit = None
            min_test = None
            for vi in v:
                if min_fit is None or vi[0] < min_fit:
                    min_test = vi[1]
                    min_fit = vi[0]
            setattr(e2, test_value, min_test)
            allexp.append(e2)
    return ExperimentGroup(allexp)


def analyse(folder, label, variant):
    print("Starting analysis")
    groups = collect_experiments_from_directory(folder, label)
    if variant == "cs":
        exp = groups[label]  # assume one group
        pd = Plotdef(title="Prediction error",
                 ylabel="log E",
                 xlabel="CPU-hours",
                 colors = {
                        "BO": "dodgerblue",
                        #"ABC ML": "dodgerblue",
                        #"ABC LM": "green",
                        #"ABC LIK": "red",
                        "ABC PM": "black",
                        #"ABC POST": "orange",
                        #"ABC MAP": "blue",
                        "GRID": "green",
                        "NELDERMEAD": "m",
                        "NELDERMEAD (PARALLEL)": "r"},
                 markers = {
                        "BO": "o",
                        #"ABC ML": "v",
                        #"ABC LM": "*",
                        #"ABC LIK": ".",
                        "ABC PM": "*",
                        #"ABC POST": ".",
                        #"ABC MAP": "^",
                        "GRID": "x",
                        "NELDERMEAD": "+",
                        "NELDERMEAD (PARALLEL)": "|"},
                 alpha=0.25,
                 figsize=(5,5),
                 errbars=True,
                 legend_loc=1)

        exp = simulate_parallel(exp, "neldermead", "MD_val", "MD_errs", "sampling_duration", 5)

        pred_errs = exp.get_value_mean_std(
                                getters= {"": lambda e: np.mean(e.MD_errs),
                                          "MED": lambda e: np.mean(e.MED_errs),
                                          #"ML": lambda e: np.mean(e.ML_errs),
                                          #"LM": lambda e: np.mean(e.LM_errs),
                                          #"LIK": lambda e: np.mean(e.lik_errs),
                                          "PM": lambda e: np.mean(e.PM_errs),
                                          #"POST": lambda e: np.mean(e.post_errs),
                                          #"MAP": lambda e: np.mean(e.MAP_errs)
                                          },
                                x_getters={"": lambda e: _get_duration(e)/3600.0,
                                           #"ABC ML": lambda e: (_get_duration(e) + e.post_duration)/3600.0,
                                           #"ABC LM": lambda e: (_get_duration(e) + e.post_duration + e.lik_sampling_duration)/3600.0,
                                           #"ABC LIK": lambda e: (_get_duration(e) + e.post_duration + e.lik_sampling_duration)/3600.0,
                                           "ABC PM": lambda e: (_get_duration(e) + e.post_duration + e.post_sampling_duration)/3600.0,
                                           #"ABC POST": lambda e: (_get_duration(e) + e.post_duration + e.post_sampling_duration)/3600.0,
                                           #"ABC MAP": lambda e: (_get_duration(e) + e.post_duration)/3600.0
                                           },
                                verbose=False)
        pred_errs = relabel(pred_errs, cs_relabeler)
        plot_trendlines(pred_errs, pd)
        #exp.plot_value_mean_std("ML value", colors, lambda e: np.exp(max(-100,e.ML_val)))
        #exp.plot_value_mean_std("MAP value", colors, lambda e: np.exp(max(-100,e.MAP_val)))
        #exp.plot_value_mean_std("Sampling duration", colors,
        #                        {"": lambda e: _get_duration(e)/3600.0,
        #                         "ABC": lambda e: (_get_duration(e) + e.post_duration)/3600.0},
        #                        relabeler=cs_relabeler)

    if variant == "irl":
        pd = Plotdef(title="Error to ground truth",
                 ylabel="RMSE",
                 hatches={"EXACT": "*", "MC50K": "O", "MC1K": "o", "ABC": "x", "ABCL": ".", "RANDOM": "\\"},
                 colors={"9x9": "burlywood", "11x11": "peru", "21x21": "chocolate", "31x31": "saddlebrown", "RANDOM": "maroon"},
                 figsize=(7,5),
                 errbars=True,
                 legend_loc="out",
                 legend_cols=3,
                 bars=15,
                 order_g=["9x9", "11x11", "21x21", "31x31", "RANDOM"],
                 order_m=["EXACT", "MC50K", "MC1K", "ABCL", "RANDOM"])
        print("Ground truth error")
        datas = dict()
        rnd = list()
        for k, v in groups.items():
            val = v.get_value_mean_std(
                              getters = {"": lambda e: e.MD_gt_err,
                                         "MED": lambda e: e.MED_gt_err,
                                         "ML": lambda e: e.ML_gt_err},
                              verbose = False)
            val = relabel(val, irl_relabeler1)
            if "RANDOM" in val.keys():
                rnd.append(val["RANDOM"])
                del val["RANDOM"]
            datas[k] = val
        datas = relabel(datas, irl_relabeler2)
        print("Aggregating random values from {}".format(rnd))
        if len(rnd) > 0:
            rand_mean = np.mean([r[0] for r in rnd])
            rand_std = np.mean([r[1] for r in rnd])
            datas["RANDOM"] = {"RANDOM": ([rand_mean], [rand_std], None, None, None, None)}
        do_ttests(datas)
        plot_barchart(datas, pd)

        pd.title = "Prediction error"
        pd.ylabel = "Discrepancy"
        pd.bars = 18
        pd.legend_cols = 4
        pd.order = ["9x9", "11x11", "21x21", "31x31"]
        print("Prediction error")
        datas = dict()
        for k, v in groups.items():
            val = v.get_value_mean_std(
                              getters = {"": lambda e: np.mean(e.MD_errs),
                                         "MED": lambda e: np.mean(e.MED_errs),
                                         "ML": {"a": lambda e: np.mean(e.ML_errs),
                                                "al": lambda e: np.mean(np.exp(e.ML_errs))}},
                              verbose = False)
            val = relabel(val, irl_relabeler1)
            datas[k] = val
        datas = relabel(datas, irl_relabeler2)
        do_ttests(datas)
        plot_barchart(datas, pd)


    #exp.print_value_mean_std("Sampling duration", lambda e: e.sampling_duration, formatter=pretty_time)
    #exp.print_value_mean_std("Minimum discrepancy value", lambda e: np.exp(e.MD_val))
    #exp.plot_value_mean_std("Minimum discrepancy value per samples", colors, lambda e: e.MD_val)
    #exp.plot_value_mean_std("Minimum discrepancy value per duration", colors,
    #                        lambda e: e.MD_val,
    #                        x_getters={"": lambda e: _get_duration(e)/3600.0,
    #                                   "ABC": lambda e: (_get_duration(e) + e.post_duration)/3600.0})


if __name__ == "__main__":
    folder = sys.argv[1]
    label = sys.argv[2]
    variant = sys.argv[3]
    analyse(folder, label, variant)

