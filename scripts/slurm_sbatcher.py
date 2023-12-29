import argparse
import os
import socket
import time
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slurm sbatcher', allow_abbrev=False)
    parser.add_argument('--file', type=str, required=True, help='File with arguments to run. Should be a list of strings, one per line')
    parser.add_argument('--at_a_time', type=int, default=-1, help='How many jobs to run at a time. If <=0, run all at once')
    parser.add_argument('--cycles', type=int, default=1, help='How many times to repeat the jobs')
    parser.add_argument('--skip_first', type=int, default=0, help='How many jobs to skip at the beginning from the \'file\'')
    parser.add_argument('--reverse', action='store_true', help='Reverse the order of the jobs')
    parser.add_argument('--name', type=str, default="mammoth", help='Name of the jobs in slurm')
    parser.add_argument('--mem', type=int, default=32, help='Memory in GB')
    parser.add_argument('--dry', action='store_true', help='Do not submit the job. Only creates the sbatch file')
    parser.add_argument('--gpus', type=int, default=1, help='How many gpus to use')
    parser.add_argument('--ddp', type=int, default=0, help='Use DistributedDataParallel. If 1, use torch.distributed.run', choices=[0, 1])
    parser.add_argument('--nodes', type=int, default=1, help='How many nodes to use. Only used if ddp=1')
    parser.add_argument('--debug', action="store_true", help='Run all jobs in debug_mode for 1 epoch')
    parser.add_argument('--timelimit', '--time', type=str, default="1-0", help='Time limit in slurm format')
    parser.add_argument('--per_job', type=int, default=1, help='How many jobs to run per slurm job')
    parser.add_argument('--excludelist', type=str, default=None, help='Nodes to exclude from the job')
    parser.add_argument('--account', '-A', type=str, default=None, help='Slurm account')
    parser.add_argument('--partition', '-p', type=str, default=None, help='Slurm partition')
    parser.add_argument('--cpus', type=int, default=8, help='How many cpus to use')
    parser.add_argument('--out', type=str, default='out', help='Output folder path')
    parser.add_argument('--err', type=str, default='err', help='Error folder path')
    parser.add_argument('--bashrc', type=str, default=None, help='Bashrc to source')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for dataloaders')

    args = parser.parse_args()

    if args.ddp:
        print("Warning: distributed stuff not yet supported in mammoth (problems with buffer synchronization). Use at your own risk!")

    with open(args.file, 'r') as f:
        all_com = f.read().splitlines()
    all_com = [x for x in all_com if not x.startswith('#') and len(x.strip())]
    all_com = all_com * args.cycles

    if args.reverse:
        all_com = all_com[::-1]
    all_com = all_com[args.skip_first:]
    if args.debug:
        sss = []
        for s in all_com:
            vv = " ".join(["--n_epochs=1" if "n_epochs=" in c else c for c in s.split()] + [" --debug_mode=1"])
            sss.append(vv)
        all_com = sss
        args.name = "debug_" + args.name

    def bbasename(path):
        return [x for x in path.split('/') if len(x)][-1]
    conf_path = os.getcwd()

    assert args.nodes == 1 or args.ddp == 1, "You can't use multiple nodes without ddp"
    assert args.ddp == 0 or args.gpus > 1, "You can't use ddp with single gpu"

    errbase, outbase = args.err, args.out

    if not os.path.exists(errbase):
        os.makedirs(errbase)
    if not os.path.exists(outbase):
        os.makedirs(outbase)

    tdir = os.getcwd()
    # os.environ['PYTHONPATH'] = f'{conf_path}'
    # os.environ['PATH'] += f':{conf_path}'
    # os.chdir(conf_path)
    len_com = math.ceil(len(all_com) / args.per_job)
    basejob_str = 'python utils/main.py' if args.ddp == 0 else f'srun python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --nproc_per_node={args.gpus} utils/main.py'

    basejob_str += (' --distributed=ddp' if args.ddp == 1 else ' --distrubuted=post_bt' if args.gpus > 1 else '')

    if args.num_workers is not None:
        basejob_str += f' --num_workers={args.num_workers}'

    if args.per_job == 1:
        jobstring = f'{basejob_str} ${{args[$SLURM_ARRAY_TASK_ID]}}'
    else:
        jobstring = basejob_str + f' &\nsleep 60s; {basejob_str}'.join([f' ${{args[$(($SLURM_ARRAY_TASK_ID * {args.per_job} + {i}))]}}' for i in range(args.per_job)])
    exclusion = '' if args.excludelist is None else '#SBATCH --exclude=' + args.excludelist
    all_com_str = "".join([f"' {s} '\n" for s in all_com]).strip()
    filec = f"""#!/bin/bash
{f"#SBATCH -p {args.partition}" if args.partition is not None else ""}
#SBATCH --job-name={args.name}
{f"#SBATCH --nodes={args.nodes}"}
#SBATCH --time={args.timelimit}
{f"#SBATCH --mem={args.mem}G" if args.mem else ""}
#SBATCH --output="{os.path.join(outbase, args.name + r'_%A_%a.out')}"
#SBATCH --error="{os.path.join(errbase, args.name + r'_%A_%a.out')}"
{f"#SBATCH -A {args.account}" if args.account is not None else ""}
#SBATCH --gres=gpu:{args.gpus}
{f"#SBATCH --cpus-per-task={args.cpus}" if args.cpus is not None else ""}
#SBATCH --array=0-{len_com-1}%{(len_com if args.at_a_time <= 0 else args.at_a_time)}
{exclusion}

{f"source {args.bashrc}" if args.bashrc is not None else ""}
export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS=1
# get random port
export MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
# get first node in slurm
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)

args=(
{all_com_str}
)
export PYTHONPATH={os.getcwd()}
cd {os.getcwd()}

""" + jobstring + '\nwait'

    outpath = 'mini_sbatch.sh'
    with open(outpath, "w") as f:
        f.write(filec)
    if args.dry:
        print(f'check {outpath}')
        exit(0)
    jobid = os.popen(f'sbatch {outpath}').read().splitlines()[-1].split()[-1].strip()
    print(jobid)
