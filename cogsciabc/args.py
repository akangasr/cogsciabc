import sys

import logging
logger = logging.getLogger(__name__)

def parse_args():
    if len(sys.argv) != 6:
        raise ValueError("arguments should be: <seed> <method> <scale> <cores> <samples>")
    args = dict()
    args["seed"] = int(sys.argv[1])
    args["method"] = sys.argv[2]
    args["scale"] = int(sys.argv[3])
    args["cores"] = int(sys.argv[4])
    args["samples"] = int(sys.argv[5])
    logger.debug("Arguments: {}".format(args))
    return args

def parse_args_grid():
    if len(sys.argv) != 7:
        raise ValueError("arguments should be: <seed> <exact> <grid_size> <n_features> <cores> <samples>")
    args = dict()
    args["seed"] = int(sys.argv[1])
    args["exact"] = bool(int(sys.argv[2]))
    args["grid_size"] = int(sys.argv[3])
    args["n_features"] = int(sys.argv[4])
    args["cores"] = int(sys.argv[5])
    args["samples"] = int(sys.argv[6])
    logger.debug("Arguments: {}".format(args))
    return args

