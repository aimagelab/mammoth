# %%
import os
import random
import sys
import json
from argparse import ArgumentParser, Namespace

if 'scripts' in os.path.dirname(os.path.abspath(__file__)):
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.getcwd()
os.chdir(mammoth_path)
sys.path.append(mammoth_path)

# %%

from numpy import array  # noqa
from torch import device  # noqa
from datasets import get_dataset


def get_accmean_from_result(result):
    dset = get_dataset(Namespace(**result))
    if f'accmean_task{dset.N_TASKS}' not in result:
        return -1
    return float(result[f'accmean_task{dset.N_TASKS}'])


if __name__ == '__main__':
    with open('scripts/reproduce.json', 'r') as f:
        runs = json.load(f)

    parser = ArgumentParser()
    parser.add_argument('--operation', type=str, default='create_list',
                        choices=['create_list', 'verify'], help='Operation to perform: '
                        '- create_list (default): create a list of commands to run on mammoth to reproduce the runs in scripts/reproduce.json '
                        '- verify: verify that the results of the runs done by the commands in data/jobs/list_reproduce_mammoth.txt are the same as the ones in scripts/reproduce.json')
    parser.add_argument('--results_dir', type=str, default='results_mammoth_reproduce', help='Directory where the results are stored')
    parser.add_argument('--jobnum', type=int, help='Identifier of the jobs to run. This will be used to filter the jobs to parse during the verify stage.')
    # parser.add_argument('--threshold', type=float, default=1e-4, help='Threshold to consider that two results are the same')
    args = parser.parse_args()

    if args.jobnum is None:
        args.jobnum = random.randint(0, 1000000)

    # runs = [
    #     {
    #         "model": ...,
    #         "cmd": ...,
    #         "result": ...|None,
    #         "setting": class-il|task-il|domain-il|general-continual,
    #     }
    # ]

    if args.operation == 'create_list':
        cmds = [r['cmd'] for r in runs]

        with open('data/jobs/list_reproduce_mammoth.txt', 'w') as f:
            f.write('\n'.join([cmd + f' --results_path={args.results_dir} --notes={args.jobnum} --non_verbose=1' for cmd in cmds]))

        print(f'Wrote {len(cmds)} jobs to data/jobs/list_reproduce_mammoth.txt')
        print(f"- Job number: {args.jobnum}")
    elif args.operation == 'verify':
        # load results from list
        old_results = [r['result'] for r in runs]

        # load results in `results_dir`
        new_results = []
        for run in runs:
            setting = run['setting']
            dataset = run['cmd'].split('--dataset')[1]
            if '=' in dataset:
                dataset = dataset.split('=')[1].split(' ')[0].strip()
            else:
                dataset = dataset.split(' ')[0].strip()
            model = run['model'].replace('-', '_')
            if not os.path.exists(os.path.join(args.results_dir, setting, dataset, model, 'logs.pyd')):
                print(f"----- Results not found for {setting}/{dataset}/{model}")
                # err_models.append(model)
                continue
            with open(os.path.join(args.results_dir, setting, dataset, model, 'logs.pyd'), 'r') as f:
                c_res = [eval(l) for l in f.readlines()]
            result = [res for res in c_res if res['notes'] == str(args.jobnum)]
            assert len(result) > 0, f"Found {len(result)} results for job {args.jobnum} in {setting}/{dataset}/{model}"

            all_res = []
            for res in result:
                all_res.append(get_accmean_from_result(res))
            new_results.append(max(all_res))

        # compare results
        bad_models = []
        for i, (old, new) in enumerate(zip(old_results, new_results)):
            if old is None:
                print(f"Results for job {i} model {runs[i]['model']}", new)
                bad_models.append(runs[i]['model'])
            else:
                print(f"Results difference for job {i}: old={old:.2f} +|+ new={new:.2f}. Model: {runs[i]['model']}")
            # if abs(old - new) > args.threshold:
                # bad_models.append(runs[i]['model'])

        if len(bad_models) == 0:
            print("All results match!")
        else:
            print(f"Results differ for {len(bad_models)} models:")
            print(bad_models)


def create_best_args():
    import json
    import yaml

    data = json.load(open('scripts/reproduce.json', 'rw'))
    all_args = {}
    for m in data:
        method = m['model']
        args = m['cmd'].strip().split()
        tmp = {}
        i = 0
        while i < len(args):
            if '=' in args[i]:
                tmp[args[i].split('=')[0]] = args[i].split('=')[1]
                i += 1
            else:
                if i < len(args) - 2:
                    if args[i + 2].startswith('-'):
                        tmp[args[i]] = args[i + 1]
                        i += 2
                    else:
                        tmp[args[i]] = [args[i + 1], args[i + 2]]
                        i += 3
                else:
                    tmp[args[i]] = args[i + 1]
                    i += 2
        all_args[method] = tmp

    for method_name, args in all_args.items():
        fname = f'models/config/{method_name}.yaml'
        if not os.path.exists(fname):
            args = {k.split('-')[-1]: str(v) for k, v in args.items()}
            dset = args['dataset']
            del args['dataset']
            if 'buffer_size' in args:
                bsize = args['buffer_size']
                del args['buffer_size']
                parsed_args = {
                    f'{dset}': {f'{bsize}': args}
                }
            else:
                parsed_args = {
                    f'{dset}': args
                }
            with open(fname, 'w') as f:
                f.write(yaml.dump(parsed_args).replace("'", ''))
