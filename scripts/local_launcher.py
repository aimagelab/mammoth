import os
import sys

if 'scripts' in os.path.dirname(os.path.abspath(__file__)):
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.getcwd()
os.chdir(mammoth_path)
sys.path.append(mammoth_path)

import functools
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool
import argparse
import signal

from utils import smart_joint

global active_jobs
global completed_jobs
global failed_jobs
active_jobs = {}
completed_jobs = {}
failed_jobs = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="file containing jobs")
    parser.add_argument("--cycles", type=int, default=1, help="number of times to run each job")
    parser.add_argument("--at_a_time", type=int, default=1, help="number of jobs to run at a time")
    parser.add_argument("--start_from", type=int, default=0, help="start from job number")
    parser.add_argument("--reverse", action="store_true", help="reverse job order")
    parser.add_argument("--preamble", type=str, help="string of arguments common to all jobs")
    parser.add_argument("--job_idxs", type=str, help="index of jobs to run (applied BEFORE `start_from`). Comma separated list of integers")
    args = parser.parse_args()

    assert args.at_a_time >= 1, "at_a_time must be at least 1"
    assert args.cycles >= 1, "cycles must be at least 1"
    assert args.start_from >= 0, "start_from must be at least 0"

    jobs_list = [l for l in open(args.file, "r").read().splitlines() if l.strip() != "" and not l.strip().startswith("#")][args.start_from:]
    job_idxs = list(range(len(jobs_list)))
    if args.job_idxs:
        job_idxs = [int(j) for j in args.job_idxs.split(",")]

    jobs_list = jobs_list * args.cycles

    if args.reverse:
        jobs_list = list(reversed(jobs_list))
    jobname = args.file.strip().split("/")[-1].split("\\")[-1].split(".")[0]
    return args, jobs_list, jobname, job_idxs


def print_progress(basepath):
    global active_jobs
    global completed_jobs
    global failed_jobs
    # clean terminal
    print("\033c", end="")

    for job_index, (jobname, pid) in active_jobs.items():
        filename = smart_joint(basepath, f'{job_index}.err')
        if not os.path.exists(filename):
            return

        print(f"Job {job_index} ({jobname}) is running with pid {pid}:")

        # show last line of error, wait for job to end
        with open(filename, "r") as err:
            try:
                last_line = err.readlines()[-1]
            except BaseException:
                last_line = ""
            print(last_line.strip())

    print("Completed jobs:" + str(len(completed_jobs)))
    print("[" + " ".join([str(job_index) for job_index, _ in completed_jobs.items()]) + "]")

    print("Failed jobs:" + str(len(failed_jobs)))
    print("[" + " ".join([str(job_index) for job_index, _ in failed_jobs.items()]) + "]")


def run_job(jobdata, basedir, jobname, log=False):
    job, index = jobdata
    global active_jobs
    global completed_jobs
    global failed_jobs
    with open(smart_joint(basedir, f'{index}.out'), "w") as out, open(smart_joint(basedir, f'{index}.err'), "w") as err:
        p = subprocess.Popen("python utils/main.py " + job, shell=True, stdout=out, stderr=err)
        active_jobs[index] = (jobname, p.pid)
        p.wait()

        # check if job failed
        if p.returncode != 0:
            failed_jobs[index] = (jobname, p.pid)
        else:
            completed_jobs[index] = (jobname, p.pid)
    del active_jobs[index]


def main():
    args, jobs_list, jobname, job_idxs = parse_args()

    print("Running {} jobs".format(len(jobs_list)))
    time.sleep(2)

    # register signal handler to kill all processes on ctrl+c
    def signal_handler(sig, frame):
        print('Killing all processes')
        if os.name == 'nt':
            for job_index, (jobname, pid) in active_jobs.items():
                os.system("taskkill /F /PID {}".format(pid))
        else:
            for job_index, (jobname, pid) in active_jobs.items():
                os.system("kill -9 {}".format(pid))
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    basedir = smart_joint("logs", jobname) + time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    print("Jobname: {}".format(jobname))
    print("Logging to {}".format(basedir))

    jobs_list = [args.preamble + " " + job for job in jobs_list]

    # create thread pool
    pool = ThreadPool(processes=args.at_a_time)
    run_fn = functools.partial(run_job, basedir=basedir, jobname=jobname)
    result = pool.map_async(run_fn, [(jobs_list[i], i) for i in job_idxs])

    # wait for all jobs to finish and print progress
    while not result._number_left == 0:
        print_progress(basedir)
        time.sleep(2)


if __name__ == '__main__':
    main()
