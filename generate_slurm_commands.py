import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 3
seed_modulo = 10000000
#methods = ["grid", "lbfgsb", "neldermead", "bo"]
methods = ["grid", "neldermead", "bo"]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "samples": [16, 81, 256, 625, 1296],
    "time": {16: "0-00:30:00",
             81: "0-00:30:00",
             256: "0-01:00:00",
             625: "0-02:00:00",
             1296: "0-04:00:00",
             2401: "0-08:00:00"},
    "mem": {16: 1000,
            81: 1000,
            256: 2000,
            625: 3000,
            1296: 5000,
            2401: 8000},
    "cores": 11,
    "scale": {16: 2,
              81: 3,
              256: 4,
              625: 5,
              1296: 6,
              2401: 7},
    },
"cogsciabc/cogsciabc/run_menumodel.py": {
    "id": "me",
    "samples": [16, 81, 256],
    "time": {16:  "1-00:00:00",
             81:  "2-00:00:00",
             256: "3-00:00:00"},
    "mem": {16:  5000,
            81:  5000,
            256: 5000},
    "cores": 21,
    "scale": {16: 2,
              81: 3,
              256: 4},
    },
}

for script, params in scripts.items():
    for method in methods:
        if params["id"] == "me" and method not in ["bo", "grid"]:
            continue
        for samples in params["samples"]:
            if params["id"] == "le" and method == "bo" and samples > 625:
                continue
            for rep in range(repl_start-1, n_replicates):
                time = params["time"][samples]
                if method == "neldermead":
                    time = "2-00:00:00"
                mem = params["mem"][samples]
                cores = params["cores"]
                scale = params["scale"][samples]
                if method in ["lbfgsb", "neldermead"]:
                    cores = 2  # not parallel
                identifier = "{}_{}_{:02d}_{:02d}"\
                        .format(params["id"], method, samples, rep+1)
                hsh = hashlib.sha224(bytearray(identifier, 'utf-8')).digest()
                seed = int.from_bytes(hsh, byteorder='big') % seed_modulo
                cmd = ["{}".format(script_name)]
                cmd.append(" -t {}".format(time))
                cmd.append(" -m {}".format(mem))
                cmd.append(" -n {}".format(cores))
                cmd.append(" -j {}".format(script))
                cmd.append(" -i {}".format(identifier))
                cmd.append(" -p {} {} {} {} {}".format(seed, method, scale, cores, samples))
                cmd.append(";")
                print("".join(cmd))
            print("")
