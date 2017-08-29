import numpy as np
import scipy as sp
from matplotlib import pyplot as pl

import logging
logger = logging.getLogger("experiment")

from cogsciabc.menumodel.features import get_feature_set

def calculate(data):
    d = get_feature_set(data)
    vals = list()
    distr = dict()
    for k in d.keys():
        f = d.get(k)
        if "task_completion_time" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 3000.0
            nbins = 8
        elif "location_of_gaze_to_target" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "proportion_of_gaze_to_target" in k:
            feature_type = "graph"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "fixation_duration" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 1000.0
            nbins = 10
        elif "saccade_duration" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 150.0
            nbins = 10
        elif "number_of_saccades" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 14.0
            nbins = 15
        elif "fixation_locations" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "length_of_skips" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        else:
            raise ValueError("Unknown feature: %s" % (k))

        bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
        if feature_type == "histogram":
            fout = [fi if fi < maxbin else maxbin+1e-10 for fi in f]
            h, e = np.histogram(fout, bins=bins)
            hnorm = h / sum(h)
        elif feature_type == "graph":
            hh, e = np.histogram(list(), bins=bins)
            hr = [0] * len(hh)
            n = [0] * len(hh)
            # assume minbin == 0, increment == 1
            for fi in f:
                hr[fi[0]] += fi[1]
                n[fi[0]] += 1
            h = list()
            for i in range(len(hr)):
                if n[i] == 0:
                    h.append(0)
                else:
                    h.append(hr[i] / float(n[i]))
            hnorm = h

        distr[k] = {
                "feature_type": feature_type,
                "f": f,
                "h": h,
                "e": e,
                "hnorm": hnorm
                }
    return distr


def plot_data(pdf, figsize, data, title):
    while type(data) is not dict:
        data = data[0]
    res = calculate(data)
    subplotrows = len(res)+1
    plotidx = 2
    fig = pl.figure(figsize=(5,20))
    title = title.replace(",", "\n")
    fig.suptitle(title, fontsize=8)
    for varname in sorted(res.keys()):
        pl.subplot(subplotrows, 1, plotidx)
        plotidx += 1
        feature_type = res[varname]["feature_type"]
        color = "g"
        bars = res[varname]["hnorm"]
        bins = res[varname]["e"]
        plot_histogram(bars, bins, color)
        vals = res[varname]["f"]
        pl.title("{}\n(m={:.2f} std={:.2f}".format(varname, np.mean(vals), np.std(vals)))
    pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    pdf.savefig()

def plot_histogram(bins, bin_edges, color="r", maxyticks=5, scalemax=None, dt=None):
    width = bin_edges[1] - bin_edges[0]
    pl.bar(bin_edges[:-1], bins, width = width, color=color)
    pl.xlim(min(bin_edges)-width*0.5, max(bin_edges)+width*0.5)
    if scalemax is None or dt is None:
        deltaticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        yticks = None
        for dt in deltaticks:
            if not max(bins) > 0.0:
                break
            yticks = np.arange(0, (int(max(bins)/dt)+2)*dt, dt)
            if len(yticks) <= maxyticks:
                pl.yticks(yticks)
                break
    else:
        yticks = np.arange(0, scalemax + dt/2.0, dt)
        pl.yticks(yticks)
    pl.show()

