# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
n_replicates = 10
seed_modulo = 1000000
methods = ["grid", "lbfgsb", "neldermead", "bo"]
scales_le = [6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 60, 80, 100]
scales_me = [6, 8, 10, 12, 14, 16]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "scales": scales_le,
    "time": {6:  "0-04:00:00",
             8:  "0-04:00:00",
             10: "0-06:00:00",
             12: "0-06:00:00",
             14: "0-12:00:00",
             16: "0-12:00:00",
             18: "1-00:00:00",
             20: "1-00:00:00",
             30: "1-00:00:00",
             40: "1-00:00:00",
             60: "1-00:00:00",
             80: "1-00:00:00",
             100: "1-00:00:00"},
    "mem": {s: 3000 for s in scales_le},
    "cores": {s: s + 1 for s in scales_le},
    "samples": {s: s*s for s in scales_le},
    },
"cogsciabc/cogsciabc/run_menumodel.py": {
    "id": "me",
    "scales": scales_me,
    "time": {6:  "5-00:00:00",
             8:  "5-00:00:00",
             10: "5-00:00:00",
             12: "5-00:00:00",
             14: "5-00:00:00",
             16: "5-00:00:00",
             18: "5-00:00:00",
             20: "5-00:00:00"},
    "mem": {s: 12000 for s in scales_me},
    "cores": {s: 2*s + 1 for s in scales_me},
    "samples": {s: s*s for s in scales_me},
    },
}

for script, params in scripts.items():
    for method in methods:
        for scale in params["scales"]:
            if method != "grid" and scale > 30:
                continue
            for rep in range(n_replicates):
                time = params["time"][scale]
                mem = params["mem"][scale]
                cores = params["cores"][scale]
                if method in ["lbfgsb", "neldermead"]:
                    cores = 2  # not parallel
                samples = params["samples"][scale]
                identifier = "{}_{}_{:02d}_{:02d}"\
                        .format(params["id"], method, scale, rep+1)
                seed = hash(identifier) % seed_modulo
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
