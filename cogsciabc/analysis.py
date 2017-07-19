import sys
import os
import re
import numpy as np

from elfie.analysis import ExperimentLog, ExperimentGroup
from elfie.utils import pretty_time

import warnings
warnings.filterwarnings("error")

import logging
logger = logging.getLogger(__name__)

def collect_experiments_from_directory(directory, script):
    print("Loading experiments from '{}'".format(directory))
    subdirs = [x[0] for x in os.walk(directory)]
    experiment_logs = list()
    if len(subdirs) == 0:
        print("Path '{}' has no subdirectories".format(directory))
        return -1
    for subdir in subdirs:
        if subdir == directory:
            continue
        m = re.match(r"{}/([A-Za-z]+)_([A-Za-z]+)_(\d+)_(\d+)".format(directory), subdir)
        if m is None:
            print("{} .. skip".format(subdir))
        else:
            script_id, method, scale, n = m.groups()
            if script_id != script:
                print("{} .. skip".format(subdir))
                continue
            scale = int(scale)
            samples = scale*scale
            path = os.path.join(subdir, "results.json")
            if not os.path.exists(path):
                print("{} .. no results".format(path))
                continue
            try:
                exp = ExperimentLog(path, method, samples)
                experiment_logs.append(exp)
                print("{} .. ok".format(path))
            except Exception as e:
                print("{} .. error".format(path))
                print(e)
    print("")
    return experiment_logs


if __name__ == "__main__":
    print("Starting analysis")
    np.seterr(all='warn')
    experiment_logs = collect_experiments_from_directory(sys.argv[1], sys.argv[2])
    exp = ExperimentGroup(experiment_logs)
    #exp.print_value_mean_std("Sampling duration", lambda e: e.sampling_duration, formatter=pretty_time)
    #exp.print_value_mean_std("Minimum discrepancy value", lambda e: np.exp(e.MD_val))
    exp.plot_value_mean_std("Minimum discrepancy value", lambda e: np.exp(e.MD_val))
    exp.plot_value_mean_std("ML value", lambda e: np.exp(e.ML_val))
    exp.plot_value_mean_std("MAP value", lambda e: np.exp(e.MAP_val))
    exp.plot_value_mean_std("Sampling duration", lambda e: e.sampling_duration)
