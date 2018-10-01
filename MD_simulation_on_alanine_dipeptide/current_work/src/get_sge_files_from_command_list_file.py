import argparse, subprocess, time
from cluster_management import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmdfile", type=str, help="file containing Python programs")
    parser.add_argument('num_jobs_per_file', type=int, help='number of jobs in each file')
    parser.add_argument('--gpu', type=int, help="whether to run on GPU")
    parser.add_argument('--folder_sge',type=str, default ='../sge_files/', help='fodler to store sge files')
    args = parser.parse_args()

    mng = cluster_management()
    mng.create_sge_files_from_a_file_containing_commands(
        command_file=args.cmdfile, num_jobs_per_file=args.num_jobs_per_file,
        folder_to_store_sge_files=args.folder_sge, run_on_gpu=args.gpu)
    return

if __name__ == '__main__':
    main()
