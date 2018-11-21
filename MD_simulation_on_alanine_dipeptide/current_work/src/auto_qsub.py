"""
This program takes a "terminal command" (should be within quotation mark) as an argument, 
generate corresponding sge file, and qsub it.
"""

from cluster_management import *
import argparse, subprocess, os

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, help="command to run")
parser.add_argument("--submit", help="submit the job", action="store_true")
parser.add_argument('--gpu', type=int, help="whether to run on GPU")
parser.add_argument("--node", type=int, default=-1)
parser.add_argument("--max_time", type=str, default='48:00:00', help='max time to run')
parser.add_argument('--use_aprun', type=int, default=1, help='use aprun in the command')
parser.add_argument("--ppn", type=int, default=2, help='number of processes per node')
args = parser.parse_args()

whether_to_qsub = args.submit
command_in_sge_file = args.command.strip()
if command_in_sge_file[-1] == '&':  # need to remove & otherwise it will not work in the cluster
    command_in_sge_file = command_in_sge_file[:-1]
server_name = subprocess.check_output(['uname', '-n']).decode("utf-8").strip()

content_for_sge_file = cluster_management.get_sge_file_content(
    [command_in_sge_file], args.gpu, max_time=args.max_time, node=args.node,
    use_aprun=args.use_aprun, ppn=args.ppn)

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
