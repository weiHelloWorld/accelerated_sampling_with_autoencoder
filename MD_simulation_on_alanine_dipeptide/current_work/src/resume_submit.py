from cluster_management import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_of_submitted_jobs_each_time", type=int, help="num_of_submitted_jobs_each_time")
parser.add_argument("--check_error", type=int, default=1, help="whether to check error for submitted jobs")
args = parser.parse_args()

a=cluster_management()
a.monitor_status_and_submit_periodically(
    num = args.num_of_submitted_jobs_each_time, monitor_mode = 'always_wait_for_submit',
    check_error_for_submitted_jobs=args.check_error
)
