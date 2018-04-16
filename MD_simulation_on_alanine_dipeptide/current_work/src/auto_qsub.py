"""
This program takes a "terminal command" (should be within quotation mark) as an argument, 
generate corresponding sge file, and qsub it.
"""

from config import *
from cluster_management import *
import argparse, subprocess, os

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, help="command to run")
parser.add_argument("--submit", help="submit the job", action="store_true")
parser.add_argument('--gpu', type=int, help="whether to run on GPU")
parser.add_argument("--node", type=int, default=-1)
args = parser.parse_args()

whether_to_qsub = args.submit
command_in_sge_file = args.command.strip()
if command_in_sge_file[-1] == '&':  # need to remove & otherwise it will not work in the cluster
    command_in_sge_file = command_in_sge_file[:-1]
server_name = subprocess.check_output(['uname', '-n']).strip()

if server_name == "alf-clustersrv.mrl.illinois.edu":
    if args.gpu:
        gpu_option_string = '#$ -l gpu=1'
    else:
        gpu_option_string = ''

    if args.node == -1:
        node_string = ""
    else:
        node_string = "#$ -l hostname=compute-0-%d" % args.node

    content_for_sge_file = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment 
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M wei.herbert.chen@gmail.com         # email address

#$ -q all.q               # queue name
#$ -l h_rt=%s       # run time (hh:mm:ss)

%s
%s
%s
echo "This job is DONE!"
exit 0
''' % (CONFIG_19, gpu_option_string, node_string, command_in_sge_file)
elif "golubh" in server_name:  # campus cluster
    content_for_sge_file = '''#!/bin/bash
#PBS -l walltime=%s
#PBS -l nodes=1:ppn=2
#PBS -l naccesspolicy=singleuser
#PBS -q alf
#PBS -V
#PBS -m ae                 # email on abort, begin, and end
#PBS -M wei.herbert.chen@gmail.com         # email address

cd $PBS_O_WORKDIR         # go to current directory
source /home/weichen9/.bashrc
%s
echo "This job is DONE!"
exit 0
''' % (CONFIG_19, command_in_sge_file)
else:
    raise Exception('server error: %s does not exist' % server_name)

folder_to_store_sge_files = '../sge_files/'

if not os.path.exists(folder_to_store_sge_files):
    subprocess.check_output(['mkdir', folder_to_store_sge_files])

assert (os.path.exists(folder_to_store_sge_files))

sge_filename = folder_to_store_sge_files + cluster_management.generate_sge_filename_for_a_command(command=command_in_sge_file)

with open(sge_filename, 'w') as sge_file:
    sge_file.write(content_for_sge_file)

if whether_to_qsub:
    subprocess.check_output(['qsub', sge_filename])
    subprocess.check_output(['rm', sge_filename])
