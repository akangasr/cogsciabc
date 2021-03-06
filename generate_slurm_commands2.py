import hashlib

# Print commands for running experiments
script_name = "./elfie/slurm/run_experiment_slurm.sh"
repl_start = 1
n_replicates = 10
seed_modulo = 10000000
script = "cogsciabc/cogsciabc/run_gridmodel.py"
cores = 11
for grid_size in [9, 11, 21, 31]:
    for method in ["exact", "sample", "sample_l", "approx", "approx_l", "random"]:
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
                elif method in ["approx", "approx_l", "sample", "sample_l"]:
                    if method == "approx":
                        ident = "a"
                    if method == "approx_l":
                        ident = "al"
                    if method == "sample":
                        ident = "s"
                    if method == "sample_l":
                        ident = "sl"
                    if n_features == 2:
                        if grid_size < 25:
                            mem = 500
                            time = "0-04:00:00"
                        else:
                            mem = 1000
                            if method == "sample_l":
                                mem = 3000
                            time = "1-00:00:00"
                    if n_features == 3:
                        if grid_size < 20:
                            mem = 1000
                            time = "0-12:00:00"
                        else:
                            mem = 2000
                            if method == "sample_l":
                                mem = 6000
                            time = "2-00:00:00"
                else:
                    ident = "r"
                    time = "0-01:00:00"
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

repl_start = 5
n_replicates = 10
script = "cogsciabc/cogsciabc/run_gridmodel_t.py"
n_features = 1
n_samples = 1
cores = 2
mem = 300
for grid_size in [5,7,9,11]:
    for method in ["exact", "sample", "sample_l", "approx"]:
        if method == "exact":
            ident = "e"
            if grid_size < 9:
                time = "0-01:00:00"
            else:
                time = "2-00:00:00"
        if method == "sample":
            ident = "s"
            time = "0-00:20:00"
        if method == "sample_l":
            ident = "sl"
            time = "0-00:20:00"
        if method == "approx":
            ident = "a"
            time = "0-00:20:00"
        for rep in range(repl_start-1, n_replicates):
            identifier = "g{}f{}t_{}_{}_{:02d}"\
                    .format(grid_size, n_features, ident, n_samples, rep+1)
            group_id = "g{}f{}t_{}_{:02d}"\
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
    print(" ")

#print("./elfie/slurm/run_experiment_slurm.sh -t 5-00:00:00 -m 10000 -n 41 -j cogsciabc/cogsciabc/run_menumodel.py -i menu_bo_1000_00 -p 123456 bo 1 41 1000;")
