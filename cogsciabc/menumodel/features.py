import numpy as np
from matplotlib import pyplot as pl

def get_feature_set(data):
    return {
        "00_task_completion_time": get_value(data, get_task_completion_time, target_present=None),
        "01_task_completion_time_target_absent": get_value(data, get_task_completion_time, target_present=False),
        "02_task_completion_time_target_present": get_value(data, get_task_completion_time, target_present=True),
        #"03_fixation_duration": get_list(data, get_fixation_durations, target_present=None),
        "03_fixation_duration_target_absent": get_list(data, get_fixation_durations, target_present=False),
        "04_fixation_duration_target_present": get_list(data, get_fixation_durations, target_present=True),
        "05_saccade_duration_target_absent": get_list(data, get_saccade_durations, target_present=False),
        "06_saccade_duration_target_present": get_list(data, get_saccade_durations, target_present=True),
        "07_number_of_saccades_target_absent": get_value(data, get_number_of_saccades, target_present=False),
        "08_number_of_saccades_target_present": get_value(data, get_number_of_saccades, target_present=True),
        "09_fixation_locations_target_absent": get_list(data, get_fixation_locations, target_present=False),
        "10_fixation_locations_target_present": get_list(data, get_fixation_locations, target_present=True),
        "11_length_of_skips_target_absent": get_list(data, get_length_of_skips, target_present=False),
        "12_length_of_skips_target_present": get_list(data, get_length_of_skips, target_present=True),
        "13_location_of_gaze_to_target": get_list(data, get_location_of_gaze_to_target, target_present=True),
        "14_proportion_of_gaze_to_target": get_value(data, get_proportion_of_gaze_to_target, target_present=True),
        }

def get_list(data, extr_fun, target_present):
    ret = list()
    for session in data["sessions"]:
        if is_valid_condition(session, target_present) == True:
            ret.extend(extr_fun(session))
    return ret

def get_value(data, extr_fun, target_present):
    return [
            extr_fun(session)
            for session in data["sessions"]
            if is_valid_condition(session, target_present) == True
            ]

def get_task_completion_time(session):
    return sum(session["duration_saccade_ms"]) + sum(session["duration_focus_ms"])

def get_location_of_gaze_to_target(session):
    ret = list()
    for gaze_location in session["gaze_location"]:
        if gaze_location == session["target_idx"]:
            ret.append(gaze_location)
    return ret

def get_proportion_of_gaze_to_target(session):
    n_gazes_to_target = 0
    n_gazes = 0
    if len(session["gaze_location"]) < 1:
        return (session["target_idx"], 0)
    for gaze_location in session["gaze_location"][:-1]: # last gaze location is end action, not a fixation
        n_gazes += 1
        if gaze_location == session["target_idx"]:
            n_gazes_to_target += 1
    if n_gazes == 0:
        return (session["target_idx"], 0)
    return (session["target_idx"], n_gazes_to_target / n_gazes)

def get_fixation_durations(session):
    if len(session["duration_focus_ms"]) < 1:
        return list()
    ret = session["duration_focus_ms"][:-1]  # assume all actions are eye fixations except last action
    if len(ret) < 1:
        # only end action
        ret = session["duration_focus_ms"]
    else:
        # adds the possible last action duration into the last fixation's duration
        ret[-1] += session["duration_focus_ms"][-1]
    return ret

def get_length_of_skips(session):
    # assume all actions are eye fixations except last action
    if len(session["gaze_location"]) < 3:
        # need at least 2 fixations and one final action
        return list()
    ret = list()
    prev_loc = session["gaze_location"][0]
    for i in range(1, len(session["gaze_location"])-1):
        cur_loc = session["gaze_location"][i]
        ret.append(abs(cur_loc - prev_loc))
        prev_loc = cur_loc
    return ret

def get_saccade_durations(session):
    return session["duration_saccade_ms"][:-1]  # assume all actions are eye fixations except last action

def get_number_of_saccades(session):
    return len(session["action"]) - 1  # assume all actions are eye fixations except last action

def get_fixation_locations(session):
    return session["action"][:-1]  # assume all actions are eye fixations except last action

def is_valid_condition(session, target_present):
    if target_present is None:
        # None -> everything is ok
        return True
    if target_present == session["target_present"]:
        return True
    return False


def plot_histogram(bins, bin_edges, color="r", maxyticks=5, scalemax=None, dt=None):
    """ Plot a histogram given bin counts and edges.
        Assumes that all bins are of equal width.
        If first and/or last bins end at infinity they
        will plotted with finite width.
    """
    width = bin_edges[1] - bin_edges[0]
    pl.bar(bin_edges[:-1], bins, width = width, color=color)
    pl.xlim(min(bin_edges), max(bin_edges))
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


def _calculate(data1, data2):
    """ ABC discrepancy function """
    distr = dict()
    for k in data1.keys():
        f1 = data1.get(k)
        f2 = data2.get(k)
        feature_type = None
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

        if feature_type == "histogram":
            bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
            f1out = [f if f < maxbin else maxbin+1e-10 for f in f1]
            f2out = [f if f < maxbin else maxbin+1e-10 for f in f2]
            h1, e1 = np.histogram(f1out, bins=bins)
            h2, e2 = np.histogram(f2out, bins=bins)
            h1norm = h1 / sum(h1)
            h2norm = h2 / sum(h2)
        elif feature_type == "graph":
            bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
            h1h, e1 = np.histogram(list(), bins=bins)
            h2h, e2 = np.histogram(list(), bins=bins)
            h1r = [0] * len(h1h)
            h2r = [0] * len(h2h)
            n1 = [0] * len(h1h)
            n2 = [0] * len(h2h)
            # assume minbin == 0, increment == 1
            for f in f1:
                h1r[f[0]] += f[1]
                n1[f[0]] += 1
            for f in f2:
                h2r[f[0]] += f[1]
                n2[f[0]] += 1
            h1 = list()
            for i in range(len(h1r)):
                if n1[i] == 0:
                    h1.append(0)
                else:
                    h1.append(h1r[i] / float(n1[i]))
            h2 = list()
            for i in range(len(h2r)):
                if n2[i] == 0:
                    h2.append(0)
                else:
                    h2.append(h2r[i] / float(n2[i]))
            h1norm = None
            h2norm = None
        else:
            raise ValueError("Unknown feature type: %s" % (feature_type))

        distr[k] = {
                "feature_type": feature_type,
                "f1": f1,
                "f2": f2,
                "h1": h1,
                "h2": h2,
                "e1": e1,
                "e2": e2,
                "h1norm": h1norm,
                "h2norm": h2norm,
                }
    return distr

def _get_avg_hist(var, histname, res):
    ret = list()
    for varname, hist in res.items():
        if varname == var:
            ret.append(np.array(hist[histname]))
    if histname in ["feature_type"]:
        return ret[0]
    if histname == "disc":
        return np.array(ret)
    if histname in ["f1", "f2"]:
        # direct ravel doesn't seem to work if sublists are not of equal length
        # lists are of unequal length because amount of raw observations may vary between realizations
        r = list()
        for l in ret:
            r.extend(l)
        return r
    return np.mean(ret, axis=0)

def plot_features(data):
    res = _calculate(data, data)
    subplotrows = len(res)+1
    subplotcols = 1
    plotidx = 1
    for varname in sorted(res.keys()):
        pl.subplot(subplotrows, subplotcols, plotidx)
        plotidx += 1
        feature_type = _get_avg_hist(varname, "feature_type", res)
        color = "g"
        if feature_type == "histogram":
            bars = _get_avg_hist(varname, "h1norm", res)
        elif feature_type == "graph":
            bars = _get_avg_hist(varname, "h1", res)
        else:
            raise ValueError("Unknown feature type: %s" % (feature_type))
        bins = _get_avg_hist(varname, "e1", res)
        plot_histogram(bars, bins, color)
        vals = _get_avg_hist(varname, "f1", res)
        pl.title("%s\n(m=%.2f std=%.2f)" % (varname, np.mean(vals), np.std(vals)))
    pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
