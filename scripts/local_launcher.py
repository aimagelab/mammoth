import functools
import os
import random
import subprocess
import sys
import time
import multiprocessing
import argparse
import signal


def smart_joint(*paths):
    return os.path.join(*paths).replace("\\", "/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="file containing jobs")
    parser.add_argument("--redundancy", type=int, default=1, help="number of times to run each job")
    parser.add_argument("--at-a-time", type=int, default=1, help="number of jobs to run at a time")
    args = parser.parse_args()
    assert args.at_a_time >= 1, "at_a_time must be at least 1"

    jobs_list = [l for l in open(args.file, "r").read().splitlines() if l.strip() != "" and not l.startswith("#")][:40] * args.redundancy
    jobname = args.file.strip().split("/")[-1].split("\\")[-1].split(".")[0]
    return args, jobs_list, jobname


def print_progress(basepath, job_index):
    filename = smart_joint(basepath, f'{job_index + 1}.err')
    if not os.path.exists(filename):
        return

    # show last line of error, wait for job to end
    with open(filename, "r") as err:
        try:
            last_line = err.readlines()[-1]
        except BaseException:
            last_line = ""
        print("\r" + last_line, end="")
    time.sleep(1 + random.random() * 2)


def run_job(jobdata, basedir, total_jobs, jobname):
    job, index = jobdata
    with open(smart_joint(basedir, f'{index + 1}.out'), "w") as out, open(smart_joint(basedir, f'{index + 1}.err'), "w") as err:
        p = subprocess.Popen("python utils/main.py " + job, shell=True, stdout=out, stderr=err)
        print("Running job {} of {} with name {} (pid: {})".format(index + 1, total_jobs, jobname, p.pid))
        p.wait()


def main():
    args, jobs_list, jobname = parse_args()

    # register signal handler to kill all processes on ctrl+c
    def signal_handler(sig, frame):
        print('Killing all processes')
        if os.name == 'nt':
            os.system("taskkill /F /T /PID {}".format(os.getpid()))
        else:
            os.system("kill -9 -1")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    basedir = smart_joint("logs", jobname)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    print("Jobname: {}".format(jobname))
    print("Logging to {}".format(basedir))

    if args.at_a_time > 1:
        pool = multiprocessing.Pool(processes=args.at_a_time)
        run_fn = functools.partial(run_job, basedir=basedir, total_jobs=len(jobs_list), jobname=jobname)
        result = pool.map_async(run_fn, [(job, i) for i, job in enumerate(jobs_list)])

        # wait for all jobs to finish and print progress
        while not result._number_left == 0:
            print_progress(basedir, len(jobs_list) - result._number_left)
    else:
        for i, job in enumerate(jobs_list):
            run_job(job, i, basedir, len(jobs_list), jobname)


if __name__ == '__main__':
    main()
