import sys
import os
import re
import fnmatch
import numpy as np

import matplotlib
import matplotlib.pyplot as pl

from elfie.analysis import ExperimentLog, ExperimentGroup
from elfie.utils import pretty_time

import warnings
warnings.filterwarnings("error")

import logging
logger = logging.getLogger(__name__)

def collect_experiments_from_directory(directory, script):
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
            print("skip (unknown) .. {}".format(subdir))
        else:
            script_id, method, samples, n = m.groups()
            if fnmatch.fnmatch(script_id, script) is False:
                print("skip ({}) .. {}".format(script_id, subdir))
                continue
            samples = int(samples)
            path = os.path.join(subdir, "results.json")
            if not os.path.exists(path):
                print("no results .. {}".format(path))
                continue
            try:
                exp = ExperimentLog(path, method, samples)
                if script_id not in experiment_logs.keys():
                    experiment_logs[script_id] = list()
                experiment_logs[script_id].append(exp)
                print("ok .. {}".format(path))
            except Exception as e:
                print("error .. {}".format(path))
                print(e)
    print("")
    groups = dict()
    for k, v in experiment_logs.items():
        groups[k] = ExperimentGroup(v)
    return groups


def _get_duration(e):
    if e.method in ["lbfgsb", "neldermead"]:
        return e.sampling_duration
    if e.method in ["grid", "bo"]:
        return e.sampling_duration * e.n_cores

def cs_relabeler(label):
    if label == "BO MED":
        return "BO"
    if label == "BO ML":
        return "ABC ML"
    if label == "BO MAP":
        return "ABC MAP"
    return label

def irl_relabeler1(label):
    if label == "A MED":
        return "APPROX"
    if label == "A ML":
        return "APPROX"
    if label == "E MED":
        return "EXACT"
    if label == "E ML":
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
    if label == "g11f2":
        return "11x11"
    if label == "g11f3":
        return "11x11"
    if label == "g13f2":
        return "13x13"
    if label == "g13f3":
        return "13x13"
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
        means, stds, x = vals
        pl.plot(x, means, marker=pd.markers[label], color=pd.colors[label], label=label)
        if pd.errbars is True:
            pl.fill_between(x, means+stds, means-stds, facecolor=pd.colors[label], alpha=pd.alpha)
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

#    bars = list()
    labels = list()
    i = 1
    for label in pd.order:
        if label not in datas.keys():
            print("Skip {}".format(label))
            continue
            i += 1
        for method in sorted(datas[label].keys()):
            labels.append("{} {}".format(label, method))
            means, stds, _ = datas[label][method]
            print(label, method)
            bar = ax.bar(ind[i], means[0], bar_width, color=pd.colors[label],
                         hatch=pd.hatches[method], edgecolor="black", zorder=3)
            if hasattr(pd, "errbars"):
                ax.errorbar(ind[i], means[0], fmt=" ",
                         yerr=stds[0], ecolor="black", capsize=5, zorder=4)
            i += 1
#        bars.append(bar)

    ax.set_ylabel(pd.ylabel, fontsize=16)
    ax.set_title(pd.title, fontsize=20)
    ax.set_xticks(ind[1:])
    ax.set_xticklabels([k for k in labels], fontsize=16)
    pl.tick_params(axis='x',which='both',bottom='off',top='off')
    pl.setp(ax.get_yticklabels(), fontsize=16)
    pl.xlim(ind[0]-0.1, ind[-1]+0.1+bar_width)

    if hasattr(pd, "ylim"):
        pl.ylim(pd.ylim)
    if hasattr(pd, "yscale"):
        pl.yscale(pd.yscale)

    #if pd.legend_loc == "in":
    #    ax.legend([b[0] for b in bars], [d["name"] for d in datas], loc=2,
    #          ncol=pd.legend_cols, fontsize=16)
    #if ed.legend_loc == "out":
    #    ax.legend([b[0] for b in bars], [d["name"] for d in datas], loc='upper center',
    #          bbox_to_anchor=(0.5, -0.1), ncol=pd.legend_cols, fontsize=16)
    pl.show()


def analyse(folder, label, variant):
    print("Starting analysis")
    groups = collect_experiments_from_directory(folder, label)
    if variant == "cs":
        exp = groups[label]  # assume one group
        pd = Plotdef(title="Prediction error",
                 ylabel="RMSE",
                 xlabel="CPU-hours",
                 colors = {
                        "BO": "skyblue",
                        "ABC ML": "dodgerblue",
                        "ABC MAP": "blue",
                        "GRID": "green",
                        "NELDERMEAD": "m"},
                 markers = {
                        "BO": "o",
                        "ABC ML": "v",
                        "ABC MAP": "^",
                        "GRID": "x",
                        "NELDERMEAD": "+"},
                 alpha=0.1,
                 figsize=(5,5),
                 errbars=True,
                 legend_loc=2)

        pred_errs = exp.get_value_mean_std(
                                getters= {"": lambda e: np.mean(e.MD_errs),
                                          "MED": lambda e: np.mean(e.MED_errs),
                                          "ML": lambda e: np.mean(e.ML_errs),
                                          "MAP": lambda e: np.mean(e.MAP_errs)},
                                x_getters={"": lambda e: _get_duration(e)/3600.0,
                                           "BO ML": lambda e: (_get_duration(e) + e.post_duration)/3600.0,
                                           "BO MAP": lambda e: (_get_duration(e) + e.post_duration)/3600.0},
                                verbose=True)
        pred_errs = relabel(pred_errs, cs_relabeler)
        plot_trendlines(pred_errs, pd)
        #exp.plot_value_mean_std("ML value", colors, lambda e: np.exp(max(-100,e.ML_val)))
        #exp.plot_value_mean_std("MAP value", colors, lambda e: np.exp(max(-100,e.MAP_val)))
        #exp.plot_value_mean_std("Sampling duration", colors,
        #                        {"": lambda e: _get_duration(e)/3600.0,
        #                         "ABC": lambda e: (_get_duration(e) + e.post_duration)/3600.0},
        #                        relabeler=cs_relabeler)

    if variant == "irl":
        pd = Plotdef(title="title",
                 ylabel="error",
                 hatches={"EXACT": "/", "APPROX": "\\", "RANDOM": ".", "7x7": "", "9x9": "", "11x11": "", "13x13": ""},
                 colors={"EXACT": "orange", "APPROX": "skyblue", "RANDOM": "gray", "7x7": "blue", "9x9": "green", "11x11": "red", "13x13": "orange"},
                 figsize=(5,5),
                 errbars=True,
                 legend_loc="in",
                 legend_cols=1,
                 bars=7,
                 order=["7x7", "9x9", "11x11", "13x13", "RANDOM"])
        print("Ground truth error")
        datas = dict()
        rnd = list()
        for k, v in groups.items():
            val = v.get_value_mean_std(
                              getters = {"": lambda e: np.mean(e.MD_gt_err),
                                         "MED": lambda e: np.mean(e.MED_gt_err),
                                         "ML": lambda e: np.mean(e.ML_gt_err),
                                         "MAP": lambda e: np.mean(e.MAP_gt_err)},
                              verbose = True)
            val = relabel(val, irl_relabeler1)
            rnd.append(val["RANDOM"])
            del val["RANDOM"]
            datas[k] = val
        datas = relabel(datas, irl_relabeler2)
        print(rnd)
        rand_mean = np.mean([r[0] for r in rnd])
        rand_std = np.mean([r[1] for r in rnd])
        datas["RANDOM"] = {"RANDOM": ([rand_mean], [rand_std], None)}
        plot_barchart(datas, pd)

        pd.bars = 10
        pd.order = ["7x7", "9x9", "11x11", "13x13"]
        print("Prediction error")
        datas = dict()
        for k, v in groups.items():
            val = v.get_value_mean_std(
                              getters = {"": lambda e: np.mean(e.MD_errs),
                                         "MED": lambda e: np.mean(e.MED_errs),
                                         "ML": lambda e: np.mean(e.ML_errs),
                                         "MAP": lambda e: np.mean(e.MAP_errs)},
                              verbose = True)
            val = relabel(val, irl_relabeler1)
            datas[k] = val
        datas = relabel(datas, irl_relabeler2)
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

