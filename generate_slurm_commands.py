# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
n_replicates = 3
seed = 10000
methods = ["grid", "uniform", "bo"]
scales = [5, 7, 10, 12, 14, 16, 18, 20]
scripts = {
"cogsciabc/cogsciabc/run_learningmodel.py": {
    "id": "le",
    "scales": scales,
    "time": {5:  "0-02:00:00",
             7:  "0-03:00:00",
             10: "0-04:00:00",
             12: "0-05:00:00",
             14: "0-06:00:00",
             16: "0-07:00:00",
             18: "0-08:00:00",
             20: "0-09:00:00"},
    "mem": {s: 1500 for s in scales},
    "cores": {s: s + 1 for s in scales},
    "samples": {s: s*s for s in scales},
    },
"cogsciabc/cogsciabc/run_menumodel.py": {
    "id": "me",
    "scales": scales,
    "time": {5:  "2-00:00:00",
             7:  "2-12:00:00",
             10: "2-12:00:00",
             12: "3-00:00:00",
             14: "3-00:00:00",
             16: "3-12:00:00",
             18: "3-12:00:00",
             20: "4-00:00:00"},
    "mem": {s: 4000 for s in scales},
    "cores": {s: 2*s + 1 for s in scales},
    "samples": {s: s*s for s in scales},
    },
}

for script, params in scripts.items():
    for method in methods:
        for scale in params["scales"]:
            for rep in range(n_replicates):
                time = params["time"][scale]
                mem = params["mem"][scale]
                cores = params["cores"][scale]
                samples = params["samples"][scale]
                identifier = "{}_{}_{:02d}_{:02d}"\
                        .format(params["id"], method, scale, rep+1)
                cmd = ["{}".format(script_name)]
                cmd.append(" -t {}".format(time))
                cmd.append(" -m {}".format(mem))
                cmd.append(" -n {}".format(cores))
                cmd.append(" -j {}".format(script))
                cmd.append(" -i {}".format(identifier))
                cmd.append(" -p {} {} {} {} {}".format(seed, method, scale, cores, samples))
                cmd.append(";")
                seed += 1
                print("".join(cmd))
