"""
this programs takes a file containing all Python programs to run as input, and 
put these programs into a workqueue, and at every instance we make sure only 
n Python programs are running

===========================
input: 

- file containing Python programs to run
- number of programs allowed to run concurrently
- time interval of checking the number of running programs
"""

import argparse, subprocess, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmdfile", type=str, help="file containing Python programs to run")
    parser.add_argument("--num", type=int, default=20, help="number of programs allowed to run concurrently")
    parser.add_argument("--interval", type=int, default=10, help="time interval of checking the number of running programs")

    args = parser.parse_args()

    command_file = args.cmdfile
    num_of_programs_allowed = args.num
    interval = args.interval

    with open(command_file, 'r') as cmdf:
        command_list = cmdf.read().split('\n')[1:]

    total_num_jobs = len(command_list)
    next_job_index = 0

    while next_job_index < total_num_jobs:
        time.sleep(interval)
        num_of_running_jobs = len(subprocess.check_output(['pidof', 'python']).split())
        if num_of_running_jobs < num_of_programs_allowed:
            if num_of_programs_allowed - num_of_running_jobs > total_num_jobs - next_job_index:
                run_programs(command_list, next_job_index, total_num_jobs)
                next_job_index = total_num_jobs
            else:
                run_programs(command_list, next_job_index, next_job_index + num_of_programs_allowed - num_of_running_jobs)
                next_job_index += num_of_programs_allowed - num_of_running_jobs


def run_programs(command_list, start_index, end_index):
    """
    run programs with index [start_index, end_index - 1]
    """
    for item in range(start_index, end_index):
        command_arg = command_list[item].split()
        if command_arg[-1] == "&":  
            command_arg = command_arg[:-1]
        print ("running command: " + str(command_arg))
        subprocess.Popen(command_arg)

    return
    

if __name__ == '__main__':
    main()
    