import argparse
import os
from pathlib import Path

from tqdm import tqdm
from multiprocessing.pool import ThreadPool

if 'scripts' in os.path.dirname(os.path.abspath(__file__)):
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.getcwd()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, help="Number of workers to use. If not specified, will use all available cores. (Recommended: n_cpus*3)")
    parser.add_argument("--limit", type=int, help="Limit the number of runs to sync")
    parser.add_argument("--reverse", action="store_true", help="Reverse the order of runs to sync")
    args = parser.parse_args()

    if args.n_workers is None:
        args.n_workers = 4 if not hasattr(os, "sched_getaffinity") else len(os.sched_getaffinity(0))
        print("'n_workers' not specified, using", args.n_workers, "workers")
    else:
        print("Using", args.n_workers, "workers")

    return args


def check_offline():
    """Checks if exists file with "offline" in name"""
    return len([f for f in os.listdir() if 'offline' in f]) > 0


def sync_run(run):
    """Syncs a single run"""
    os.system(f"wandb sync {run} >>synced.log 2>>err.log")


if __name__ == "__main__":
    args = parse_args()
    os.chdir(mammoth_path)

    if check_offline():
        runlist = [f for f in os.listdir() if "offline" in f.lower()]
    else:
        runlist = [os.path.join("wandb", f) for f in os.listdir('wandb') if "offline" in f.lower()]

    if args.reverse:
        runlist = runlist[::-1]

    if args.limit is not None:
        runlist = runlist[:args.limit]
        print("Limiting to", args.limit, "runs")

    print(len(runlist), "runs to sync")

    # delete file synced.log if exists
    if Path("synced.log").exists():
        Path("synced.log").unlink()

    # delete file err.log if exists
    if Path("err.log").exists():
        Path("err.log").unlink()

    # sync all runs in multiple threads and log tqdm
    with ThreadPool(args.n_workers) as p:
        r = list(tqdm(p.imap(sync_run, runlist), total=len(runlist)))

    # check if there are any errors in err.log
    if Path("err.log").exists():
        with open("err.log", "r") as f:
            if f.read():
                print("Error in syncing, check err.log")
                exit(1)
            else:
                print("No error in syncing")
                exit(0)

    exit(0)
