import os
import random
import sys
import json
from argparse import ArgumentParser

if 'scripts' in os.path.dirname(os.path.abspath(__file__)):
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.getcwd()
os.chdir(mammoth_path)
sys.path.append(mammoth_path)

if __name__ == '__main__':
    with open('scripts/reproduce.json', 'r') as f:
        runs = json.load(f)

    parser = ArgumentParser()
    parser.add_argument('--operation', type=str, default='create_list',
                        choices=['create_list', 'verify'], help='Operation to perform: '
                        '- create_list (default): create a list of commands to run on mammoth to reproduce the runs in scripts/reproduce.json '
                        '- verify: verify that the results of the runs done by the commands in data/jobs/list_reproduce_mammoth.txt are the same as the ones in scripts/reproduce.json')
    parser.add_argument('--results_dir', type=str, default='data/results_mammoth_reproduce', help='Directory where the results are stored')
    parser.add_argument('--jobnum', type=int, help='Identifier of the jobs to run. This will be used to filter the jobs to parse during the verify stage.')
    parser.add_argument('--threshold', type=float, default=1e-4, help='Threshold to consider that two results are the same')
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
            f.write('\n'.join([cmd + f' --results_path={args.results_dir} --notes={args.jobnum}' for cmd in cmds]))

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
            model = run['model']
            with open(os.path.join(args.results_dir, setting, dataset, model, 'logs.pyd'), 'r') as f:
                c_res = [eval(l) for l in f.readlines()]
            result = [res for res in c_res if res['notes'] == args.jobnum]
            assert len(result) == 1, f"Found {len(result)} results for job {args.jobnum} in {setting}/{dataset}/{model}"

            c_res = result[0]
            last_task = max([int(v.split('accmean_task')[-1]) for v in c_res.keys() if 'accmean_task' in v])

            new_results.append(float(c_res[f'accmean_task{last_task}']))

        # compare results
        bad_models = []
        for i, (old, new) in enumerate(zip(old_results, new_results)):
            if abs(old - new) > args.threshold:
                print(f"Results differ for job {i}: {old} != {new}. Model: {runs[i]['model']}")
                bad_models.append(runs[i]['model'])

        if len(bad_models) == 0:
            print("All results match!")
        else:
            print(f"Results differ for {len(bad_models)} models:")
            print(bad_models)
