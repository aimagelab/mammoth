import os
import itertools
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Prepare grid')
parser.add_argument('--begin_string', type=str, default="", help='Initial string common for all experiments')
parser.add_argument('--job_folder', type=str, default="data/jobs/", help='Folder to save the job list')
args = parser.parse_args()

grid_combinations = [
    {
        'name': 'experiment_name',
        'combos': {
            'lr': [0.01, 0.3, 0.05],
            'buffer_size': [500],
            'model': ['er'],
            'dataset': ['seq-cifar10']
        },
    },
]

configs = []
all_configs = []

for experiment in grid_combinations:
    filenam, combos = experiment['name'], experiment['combos']
    configs = list(itertools.product(*combos.values()))

    print(filenam, len(configs), 'items')

    begin = args.begin_string
    folder = args.job_folder
    os.makedirs(folder, exist_ok=True, mode=0o777)

    clines = 0
    print(f'{folder}list_{filenam}.txt')
    with open(f'{folder}list_{filenam}.txt', 'w') as f:
        for c in configs:
            ll = begin
            for k, v in zip(combos.keys(), c):
                if v is None:
                    continue
if isinstance(k,                 if)                    for i in range(len(k)):
                        ll += f" --{k[i]}={v[i]}"
                else:
                    ll += f" --{k}={v}"
            f.write(ll +'\n')
            all_configs.append(ll)

            clines += 1

    print(f"Total ({filenam}):", clines)

print(f'{folder}list_all_grid.txt')
clines = 0
with open(f'{folder}list_all_grid.txt', 'w') as f:
    for ll in all_configs:
        f.write(ll + '\n')
        clines += 1

print("Total (all):", clines)
print('')
