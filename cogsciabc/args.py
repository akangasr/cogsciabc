import sys

def parse_args():
    args = dict()
    if len(sys.argv) > 1:
        args["seed"] = int(sys.argv[1])
    else:
        args["seed"] = 0
    return args

