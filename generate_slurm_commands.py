import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 10
seed_modulo = 10000000
#methods = ["grid", "lbfgsb", "neldermead", "bo"]
methods = ["grid", "neldermead", "bo"]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "samples": [16, 81, 256, 625, 1296, 2401],
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
    "samples": [27, 64, 125, 216, 343, 512],
    "time": {27:  "1-00:00:00",
             64:  "1-00:00:00",
             125: "2-00:00:00",
             216: "3-00:00:00",
             343: "4-00:00:00",
             512: "5-00:00:00"},
    "mem": {27:  4000,
            64:  4000,
            125: 4000,
            216: 5000,
            343: 5000,
            512: 6000},
    "cores": 21,
    "scale": {27: 3,
              64: 4,
              125: 5,
              216: 6,
              343: 7,
              512: 8},
    },
}

for script, params in scripts.items():
    for method in methods:
        if params["id"] == "me" and method not in ["bo", "grid"]:
            continue
        for samples in params["samples"]:
            if params["id"] == "le" and method != "grid" and samples > 1000:
                continue
            for rep in range(repl_start-1, n_replicates):
                time = params["time"][samples]
                cores = params["cores"]
                if True or method in ["lbfgsb", "neldermead"]:
                    cores = 2  # not parallel
                    if samples < 100:
                        time = "0-04:00:00"
                    elif samples < 300:
                        time = "0-08:00:00"
                    elif samples < 1000:
                        time = "1-00:00:00"
                    elif samples < 1500:
                        time = "1-12:00:00"
                    else:
                        time = "2-00:00:00"
                mem = params["mem"][samples]
                scale = params["scale"][samples]
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
