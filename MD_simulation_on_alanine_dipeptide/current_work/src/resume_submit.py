from cluster_management import *

import sys

num_of_submitted_jobs_each_time = int(sys.argv[1])

a=cluster_management()
a.monitor_status_and_submit_periodically(num = num_of_submitted_jobs_each_time, monitor_mode = 'always_wait_for_submit')
