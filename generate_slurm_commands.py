import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 15
seed_modulo = 10000000
#methods = ["grid", "lbfgsb", "neldermead", "bo"]
methods = ["grid", "neldermead", "bo"]
scales_le = [10, 15, 20, 25, 30, 35, 40, 45]
scales_ch = [10, 20, 30, 40, 50, 60]
scales_me = [6, 8, 10, 12, 14]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "scales": scales_le,
    "time": {10: "0-00:30:00",
             15: "0-01:00:00",
             20: "0-01:30:00",
             25: "0-02:00:00",
             30: "0-02:30:00",
             35: "0-03:00:00",
             40: "0-03:00:00",
             45: "0-05:00:00"},
    "mem": {10: 1000,
            15: 1000,
            20: 1000,
            25: 2000,
            30: 3000,
            35: 4000,
            40: 5000,
            45: 6000},
    "cores": {s: 11 for s in scales_le},
    "samples": {s: s*s for s in scales_le},
    },
"cogsciabc/cogsciabc/run_menumodel.py": {
    "id": "me",
    "scales": scales_me,
    "time": {6:  "1-12:00:00",
             8:  "2-00:00:00",
             10: "2-00:00:00",
             12: "3-00:00:00",
             14: "3-00:00:00"},
    "mem": {6:  5000,
            8:  5000,
            10: 5000,
            12: 5000,
            14: 5000},
    "cores": {s: 21 for s in scales_me},
    "samples": {s: s*s for s in scales_me},
    },
}

for script, params in scripts.items():
    for method in methods:
        if params["id"] == "me" and method not in ["bo", "grid"]:
            continue
        for scale in params["scales"]:
            if params["id"] == "le" and method == "bo" and scale > 35:
                continue
            for rep in range(repl_start-1, n_replicates):
                time = params["time"][scale]
                if method == "neldermead":
                    time = "1-12:00:00"
                mem = params["mem"][scale]
                cores = params["cores"][scale]
                if method in ["lbfgsb", "neldermead"]:
                    cores = 2  # not parallel
                samples = params["samples"][scale]
                identifier = "{}_{}_{:02d}_{:02d}"\
                        .format(params["id"], method, scale, rep+1)
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
