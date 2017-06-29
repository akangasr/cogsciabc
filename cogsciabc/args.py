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

