import copy, pickle, re, os, time, subprocess, datetime, itertools, hashlib, glob, numpy as np

class cluster_management(object):
    def __init__(self):
        return

    @staticmethod
    def get_server_and_user():
        server = subprocess.check_output(['uname', '-n']).decode("utf-8").strip()
        user = subprocess.check_output('echo $HOME', shell=True).decode("utf-8").strip().split('/')[-1]
        return server, user

    @staticmethod
    def get_sge_file_content(command_list, gpu, max_time, node=-1,
                             num_nodes=None,    # blue waters only
                             use_aprun=True,   # blue waters only
                             ppn=2):
        assert (isinstance(command_list, list))
        temp_commands = []
        for item in command_list:
            item = item.strip()
            if item[-1] != '&':
                item += ' &'         # to run multiple jobs in a script
            temp_commands.append(item)
        assert (len(temp_commands) == len(command_list))
        server_name, _ = cluster_management.get_server_and_user()
        if 'alf' in server_name:
            gpu_option_string = '#$ -l gpu=1' if gpu else ''
            node_string = "" if node == -1 else "#$ -l hostname=compute-0-%d" % node

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
wait       # to wait for all jobs to finish
echo "This job is DONE!"
exit 0
''' % (max_time, gpu_option_string, node_string, '\n'.join(temp_commands))
        elif "golubh" in server_name:  # campus cluster
            content_for_sge_file = '''#!/bin/bash
#PBS -l walltime=%s
#PBS -l nodes=1:ppn=%d
#PBS -l naccesspolicy=singleuser
#PBS -q alf
#PBS -V
#PBS -m ae                 # email on abort, begin, and end
#PBS -M wei.herbert.chen@gmail.com         # email address

cd $PBS_O_WORKDIR         # go to current directory
source /home/weichen9/.bashrc
%s
wait       # to wait for all jobs to finish
echo "This job is DONE!"
exit 0
''' % (max_time, ppn, '\n'.join(temp_commands))
        elif "h2ologin" in server_name or 'nid' in server_name:  # Blue Waters
            if num_nodes is None:
                num_nodes = len(command_list)
            node_type = ':xk' if gpu else ''
            if use_aprun:
                temp_commands = ['aprun -n1 ' + item for item in temp_commands]
            content_for_sge_file = '''#!/usr/bin/zsh
#PBS -l walltime=%s
#PBS -l nodes=%d:ppn=%d%s
#PBS -m ae   
#PBS -M wei.herbert.chen@gmail.com    
#PBS -A batp

. /etc/zsh.zshrc.local
source /u/sciteam/chen21/.zshrc
cd $PBS_O_WORKDIR
export PMI_NO_FORK=1
export PMI_NO_PREINITIALIZE=1
module unload bwpy
module load bwpy/2.0.0-pre1
source /u/sciteam/chen21/.myPy3/bin/activate
%s
wait       # to wait for all jobs to finish
echo "This job is DONE!"
exit 0
''' % (max_time, num_nodes, ppn, node_type, '\n'.join(temp_commands))
        else:
            raise Exception('server error: %s does not exist' % server_name)
        return content_for_sge_file

    @staticmethod
    def generate_sge_filename_for_a_command(command):
        sge_filename = command.split('>')[0]
        for item in ('"', '&', 'python', "'", '\\'):
            sge_filename = sge_filename.replace(item, '')
        for item in (' ', '..', '/', '--', ':'):
            sge_filename = sge_filename.replace(item, '_')
        sge_filename = sge_filename.strip() + '.sge'
        sge_filename = re.sub('_+', '_', sge_filename)
        if len(sge_filename) > 255:  # max length of file names in Linux
            temp = hashlib.md5()
            temp.update(sge_filename)
            sge_filename = "h_" + temp.hexdigest() + sge_filename[-200:]
        return sge_filename

    @staticmethod
    def create_sge_files_from_a_file_containing_commands(
            command_file, num_jobs_per_file=1, folder_to_store_sge_files='../sge_files/', run_on_gpu = False):
        with open(command_file, 'r') as commmand_file:
            commands_to_run = commmand_file.readlines()
            commands_to_run = [x.strip() for x in commands_to_run]
            commands_to_run = [x for x in commands_to_run if x != ""]
            cluster_management.create_sge_files_for_commands(
                commands_to_run, num_jobs_per_file=num_jobs_per_file,
                folder_to_store_sge_files=folder_to_store_sge_files,
                run_on_gpu=run_on_gpu
            )

        return commands_to_run

    @staticmethod
    def create_sge_files_for_commands(list_of_commands_to_run,
                                      num_jobs_per_file=1,   # may have more than 1 jobs in each file, for efficiency of scheduling
                                      folder_to_store_sge_files = '../sge_files/',
                                      run_on_gpu = False, ppn=2):
        if folder_to_store_sge_files[-1] != '/':
            folder_to_store_sge_files += '/'
        if not os.path.exists(folder_to_store_sge_files):
            subprocess.check_output(['mkdir', folder_to_store_sge_files])
        sge_file_list = []
        num_files = int(np.ceil(float(len(list_of_commands_to_run)) / float(num_jobs_per_file)))
        for index in range(num_files):
            item_command_list = list_of_commands_to_run[index * num_jobs_per_file: (index + 1) * num_jobs_per_file]
            sge_filename = cluster_management.generate_sge_filename_for_a_command(item_command_list[0])  # use first command to generate file name
            sge_filename = folder_to_store_sge_files + sge_filename
            sge_file_list.append(sge_filename)

            content_for_sge_files = cluster_management.get_sge_file_content(
                item_command_list, gpu=run_on_gpu, max_time='24:00:00', ppn=ppn)
            with open(sge_filename, 'w') as f_out:
                f_out.write(content_for_sge_files)
                f_out.write("\n")
        assert (len(sge_file_list) == num_files)
        return sge_file_list

    @staticmethod
    def get_num_of_running_jobs():
        _, user = cluster_management.get_server_and_user()
        output = subprocess.check_output(['qstat', '-u', user]).decode("utf-8")
        all_entries = output.strip().split('\n')[2:]   # remove header
        all_entries = [item for item in all_entries if user in item]        # remove unrelated lines
        all_entries = [item for item in all_entries if (not item.strip().split()[4] == 'dr')]   # remove job in "dr" state
        num_of_running_jobs = len(all_entries)
        # print('checking number of running jobs = %d\n' % num_of_running_jobs)
        return num_of_running_jobs

    @staticmethod
    def submit_sge_jobs_and_archive_files(job_file_lists,
                                          num  # num is the max number of jobs submitted each time
                                          ):
        dir_to_archive_files = '../sge_files/archive/'

        if not os.path.exists(dir_to_archive_files):
            os.makedirs(dir_to_archive_files)

        assert(os.path.exists(dir_to_archive_files))
        sge_job_id_list = []
        for item in job_file_lists[0:num]:
            output_info = subprocess.check_output(['qsub', item]).decode("utf-8").strip()
            sge_job_id_list.append(cluster_management.get_job_id_from_qsub_output(output_info))
            print('submitting ' + str(item))
            subprocess.check_output(['mv', item, dir_to_archive_files]) # archive files
        return sge_job_id_list

    @staticmethod
    def get_job_id_from_qsub_output(output_info):
        server, _ = cluster_management.get_server_and_user()
        if 'alf' in server:
            result = output_info.strip().split(' ')[2]
        elif "h2ologin" in server or 'nid' in server:
            result = output_info.strip().split('\n')[-1]
            assert (result[-3:] == '.bw')
            result = result[:-3]
            assert (len(result) == 7)
        else: raise Exception('unknown server')
        return result

    @staticmethod
    def submit_a_single_job_and_wait_until_it_finishes(job_sge_file):
        job_id = cluster_management.submit_sge_jobs_and_archive_files([job_sge_file], num=1)[0]
        print("job = %s, job_id = %s" % (job_sge_file, job_id))
        while cluster_management.is_job_on_cluster(job_id):
            time.sleep(10)
        print("job (id = %s) done!" % job_id)
        return job_id

    @staticmethod
    def run_a_command_and_wait_on_cluster(command, ppn=2):
        print('running %s on cluster' % command)
        sge_file = cluster_management.create_sge_files_for_commands([command], ppn=ppn)[0]
        id = cluster_management.submit_a_single_job_and_wait_until_it_finishes(sge_file)
        return id

    @staticmethod
    def get_output_and_err_with_job_id(job_id):
        temp_file_list = subprocess.check_output(['ls']).decode("utf-8").strip().split('\n')
        out_file = list([x for x in temp_file_list if '.sge.o' + job_id in x])[0]
        err_file = list([x for x in temp_file_list if '.sge.e' + job_id in x])[0]
        return out_file, err_file

    @staticmethod
    def get_sge_files_list():
        result = [x for x in subprocess.check_output(['ls', '../sge_files']).decode("utf-8").split('\n') if x[-3:] == "sge"]
        result = ['../sge_files/' + x for x in result]
        return result

    @staticmethod
    def submit_new_jobs_if_there_are_too_few_jobs(num):
        if cluster_management.get_num_of_running_jobs() < num:
            job_list = cluster_management.get_sge_files_list()
            job_id_list = cluster_management.submit_sge_jobs_and_archive_files(job_list,
                                    num - cluster_management.get_num_of_running_jobs())
        else:
            job_id_list = []
        return job_id_list

    @staticmethod
    def monitor_status_and_submit_periodically(num,
                                               num_of_running_jobs_when_allowed_to_stop = 0,
                                               monitor_mode = 'normal',  # monitor_mode determines whether it can go out of first while loop
                                               check_error_for_submitted_jobs=True):
        if monitor_mode == 'normal':
            min_num_of_unsubmitted_jobs = 0
        elif monitor_mode == 'always_wait_for_submit':
            min_num_of_unsubmitted_jobs = -1
        else:
            raise Exception('monitor_mode not defined')

        submitted_job_id = []
        num_of_unsubmitted_jobs = len(cluster_management.get_sge_files_list())
        # first check if there are unsubmitted jobs
        while len(submitted_job_id) > 0 or num_of_unsubmitted_jobs > min_num_of_unsubmitted_jobs:
            time.sleep(10)
            if check_error_for_submitted_jobs:
                cluster_management.get_sge_dot_e_files_in_current_folder_and_handle_jobs_not_finished_successfully()
            try:
                temp_submitted_job_id = cluster_management.submit_new_jobs_if_there_are_too_few_jobs(num)
                submitted_job_id += temp_submitted_job_id
                submitted_job_id = list([x for x in submitted_job_id if cluster_management.is_job_on_cluster(x)])   # remove finished id of finished jobs
                print("submitted_job_id = %s" % str(submitted_job_id))
                num_of_unsubmitted_jobs = len(cluster_management.get_sge_files_list())
            except:
                print("not able to submit jobs!\n")

        # then check if all jobs are done (not really, since there could be multiple cluster_management running)
        while cluster_management.get_num_of_running_jobs() > num_of_running_jobs_when_allowed_to_stop:
            time.sleep(10)
        return

    @staticmethod
    def is_job_on_cluster(job_sgefile_name):    # input could be sge file name or job id
        server, user = cluster_management.get_server_and_user()
        if 'alf' in server:
            output = subprocess.check_output(['qstat', '-r']).decode("utf-8")
            result = job_sgefile_name in output
        elif "h2ologin" in server or 'nid' in server:
            output = subprocess.check_output(['qstat', '-u', user, '-f', '-x']).decode("utf-8")   # output in xml format, make sure long file name is displayed in one line
            result = False
            for item in output.split('<Job>')[1:]:
                if (job_sgefile_name in item) and (not '<job_state>C</job_state>' in item):  # ignore completed jobs
                    result = True
        else: raise Exception('unknown server')
        return result

    @staticmethod
    def check_whether_job_finishes_successfully(job_sgefile_name, latest_version = True):
        """
        return value:
        0: finishes successfully
        3: not finished
        1: finishes with exception
        2: aborted due to time limit or other reason
        -1: job does not exist
        """
        job_finished_message = 'This job is DONE!'
        # first we check whether the job finishes
        if cluster_management.is_job_on_cluster(job_sgefile_name):
            return 3  # not finished
        else:
            all_files_in_this_dir = sorted(glob.glob('*'))
            out_file_list = [x for x in all_files_in_this_dir if job_sgefile_name + ".o" in x]
            err_file_list = [x for x in all_files_in_this_dir if job_sgefile_name + ".e" in x]

            if len(out_file_list) == 0 or len(err_file_list) == 0:
                print("%s does not exist" % job_sgefile_name)
                return -1  

            if latest_version:   # check output/error information for the latest version, since a job could be submitted multiple times
                job_serial_number_list = [int(x.split('.sge.o')[1]) for x in out_file_list]
                job_serial_number_of_latest_version = max(job_serial_number_list)
                latest_out_file = filter(lambda x: str(job_serial_number_of_latest_version) in x, out_file_list)[0]
                latest_err_file = filter(lambda x: str(job_serial_number_of_latest_version) in x, err_file_list)[0]
                with open(latest_out_file, 'r') as out_f:
                    out_content = [item.strip() for item in out_f.readlines()]

                with open(latest_err_file, 'r') as err_f:
                    err_content = [item.strip() for item in err_f.readlines()]
                    err_content = [x for x in err_content if "Traceback (most recent call last)" in x]

                if (job_finished_message in out_content) and (len(err_content) != 0):
                    print("%s ends with exception" % job_sgefile_name)
                    return 1  
                elif not job_finished_message in out_content:
                    print("%s aborted due to time limit or other reason" % job_sgefile_name)
                    return 2  
                else:
                    print("%s finishes successfully" % job_sgefile_name)
                    return 0  
            else: return

    @staticmethod
    def handle_jobs_not_finished_successfully_and_archive(job_sgefile_name_list, latest_version=True):
        dir_to_archive_files = '../sge_files/archive/'
        folder_to_store_sge_files='../sge_files/'
        if not os.path.exists(dir_to_archive_files):
            subprocess.check_output(['mkdir', dir_to_archive_files])

        if not os.path.exists(folder_to_store_sge_files):
            subprocess.check_output(['mkdir', folder_to_store_sge_files])
            
        for item in job_sgefile_name_list:
            status_code = cluster_management.check_whether_job_finishes_successfully(item, latest_version)
            if status_code in (1, 2):
                if os.path.isfile(dir_to_archive_files + item):
                    print("restore sge_file: %s" % item)
                    subprocess.check_output(['cp', dir_to_archive_files + item, folder_to_store_sge_files])
                    assert (os.path.exists(folder_to_store_sge_files + item))
                else:
                    print("%s not exists in %s" % (item, dir_to_archive_files))
            
            if status_code in (0, 1, 2):  # archive .o/.e files for finished jobs
                print("archive .o/.e files for %s" % item)
                all_files_in_this_dir = sorted(glob.glob('*'))
                temp_dot_o_e_files_for_this_item = [x for x in all_files_in_this_dir if (item + '.o' in x) or (item + '.e' in x)]
                for temp_item_o_e_file in temp_dot_o_e_files_for_this_item:
                    subprocess.check_output(['mv', temp_item_o_e_file, dir_to_archive_files])
        return

    @staticmethod
    def get_sge_dot_e_files_in_current_folder_and_handle_jobs_not_finished_successfully(latest_version=True):
        sge_e_files = glob.glob('*.sge.e*')
        sge_files = [item.split('.sge')[0] + '.sge' for item in sge_e_files]
        sge_files = list(set(sge_files))
        # print "sge_files = %s" % str(sge_files)
        cluster_management.handle_jobs_not_finished_successfully_and_archive(sge_files, latest_version)
        return
