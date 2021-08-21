import os


def run_serial(ks, param_combinations, path):
    '''run each parameter combination in serial
    '''
    for i in range(len(param_combinations)):
        param_str = f'python {path} '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print(f'running: {param_str}\n\t({i}/{len(param_combinations)})')
        os.system(param_str)


def run_parallel(ks, param_combinations, path, partition='low'):
    '''run each parameter combination in parallel (requires slurmpy package)
    '''
    from slurmpy import Slurm
    s = Slurm("fit_mog", {"partition": partition, "time": "4-0"})

    for i in range(len(param_combinations)):
        param_str = f'module load python; python3 {path} '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print(f'scheduled: {param_str}\n\t({i}/{len(param_combinations)})')
        s.run(param_str)
