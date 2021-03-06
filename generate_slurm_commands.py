import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 10
seed_modulo = 10000000
methods = ["grid", "neldermead", "bo"]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "samples": {"grid": [16, 81, 256, 625, 1296, 2401],
                "neldermead": [20, 30, 50, 110, 200, 300, 400, 500, 700],
                "bo": [25, 75, 150, 250, 350, 450, 550]},
    "cores": 2,
    "scale": {16: 2,
              81: 3,
              256: 4,
              625: 5,
              1296: 6,
              2401: 7},
    },
"cogsciabc/cogsciabc/run_menumodel.py": {
    "id": "me",
    "samples": {"grid": [27, 64, 125, 216, 343, 512],
                "neldermead": [10, 20, 30, 40, 50],
                "bo": [20, 60, 120, 240, 420]},
    "scale": {27: 3,
              64: 4,
              125: 5,
              216: 6,
              343: 7,
              512: 8},
    },
}

for script, params in scripts.items():
    for method, samplesl in params["samples"].items():
        for samples in samplesl:
            for rep in range(repl_start-1, n_replicates):
                if params["id"] == "me":
                    cores = 21
                    mem = 6000
                    if samples < 80:
                        time = "0-14:00:00"
                    elif samples < 150:
                        time = "1-00:00:00"
                    elif samples < 300:
                        time = "2-00:00:00"
                    else:
                        time = "4-00:00:00"
                    if method == "neldermead":
                        time = "5-00:00:00"
                        cores = 2
                if params["id"] == "le":
                    cores = 2
                    mem = 3000
                    if samples < 100:
                        time = "0-04:00:00"
                    elif samples < 300:
                        time = "0-08:00:00"
                    elif samples < 1000:
                        time = "1-00:00:00"
                    elif samples < 1000:
                        time = "1-12:00:00"
                    else:
                        time = "2-00:00:00"
                if samples in params["scale"]:
                    scale = params["scale"][samples]
                else:
                    scale = 1
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
