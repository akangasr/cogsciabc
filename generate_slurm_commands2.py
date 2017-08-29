import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 15
seed_modulo = 10000000
script = "cogsciabc/cogsciabc/run_gridmodel.py"
cores = 11
for grid_size in [7, 9, 11, 21, 31]:
    for method in ["exact", "approx", "random"]:
        for n_features in [2,3]:
            if grid_size > 11 and method == "exact":
                continue
            for rep in range(repl_start-1, n_replicates):
                if n_features == 2:
                    n_samples = 200
                if n_features == 3:
                    n_samples = 600
                if method == "exact":
                    ident = "e"
                    mem = 500
                    if n_features == 2:
                        time = "1-00:00:00"
                    if n_features == 3:
                        time = "3-00:00:00"
                elif "approx" in method:
                    ident = "a"
                    mem = 500
                    if n_features == 2:
                        if grid_size < 20:
                            time = "0-04:00:00"
                        else:
                            time = "1-00:00:00"
                    if n_features == 3:
                        if grid_size < 20:
                            time = "0-12:00:00"
                        else:
                            time = "2-00:00:00"
                else:
                    ident = "r"
                    time = "0-00:20:00"
                    mem = 300
                identifier = "g{}f{}_{}_{}_{:02d}"\
                        .format(grid_size, n_features, ident, n_samples, rep+1)
                group_id = "g{}f{}_{}_{:02d}"\
                        .format(grid_size, n_features, n_samples, rep+1)
                hsh = hashlib.sha224(bytearray(group_id, 'utf-8')).digest()
                seed = int.from_bytes(hsh, byteorder='big') % seed_modulo
                cmd = ["{}".format(script_name)]
                cmd.append(" -t {}".format(time))
                cmd.append(" -m {}".format(mem))
                cmd.append(" -n {}".format(cores))
                cmd.append(" -j {}".format(script))
                cmd.append(" -i {}".format(identifier))
                cmd.append(" -p {} {} {} {} {} {}".format(seed, method, grid_size, n_features, cores, n_samples))
                cmd.append(";")
                print("".join(cmd))
        print("")

#print("./elfie/slurm/run_experiment_slurm.sh -t 5-00:00:00 -m 10000 -n 41 -j cogsciabc/cogsciabc/run_menumodel.py -i menu_bo_1000_00 -p 123456 bo 1 41 1000;")
