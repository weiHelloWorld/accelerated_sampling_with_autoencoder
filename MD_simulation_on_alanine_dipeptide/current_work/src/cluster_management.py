from config import *
import copy, pickle, re, os, time, subprocess, datetime, itertools, hashlib

class cluster_management(object):
    def __init__(self):
        return

    @staticmethod
    def create_sge_files_from_a_file_containing_commands(command_file, folder_to_store_sge_files='../sge_files/', run_on_gpu = False):
        with open(command_file, 'r') as commmand_file:
            commands_to_run = commmand_file.readlines()
            commands_to_run = map(lambda x: x.strip(), commands_to_run)
            commands_to_run = filter(lambda x: x != "", commands_to_run)
            cluster_management.create_sge_files_for_commands(commands_to_run, folder_to_store_sge_files, run_on_gpu)

        return commands_to_run

    @staticmethod
    def create_sge_files_for_commands(list_of_commands_to_run, folder_to_store_sge_files = '../sge_files/', run_on_gpu = False):
        if run_on_gpu:
            gpu_option_string = '#$ -l gpu=1'
        else:
            gpu_option_string = ''

        sge_file_list = []
        for item in list_of_commands_to_run:
            item = item.strip()
            if item[-1] == '&':  # need to remove & otherwise it will not work in the cluster
                item = item[:-1]

            if folder_to_store_sge_files[-1] != '/':
                folder_to_store_sge_files += '/'

            if not os.path.exists(folder_to_store_sge_files):
                subprocess.check_output(['mkdir', folder_to_store_sge_files])

            sge_filename = item.replace(' ', '_').replace('..', '_').replace('/','_')\
                .replace('&', '').replace('--', '_').replace('\\','') + '.sge'
            sge_filename = re.sub('_+', '_', sge_filename)

            if len(sge_filename) > 255:    # max length of file names in Linux
                temp = hashlib.md5()
                temp.update(sge_filename)
                sge_filename = "h_" + temp.hexdigest() + sge_filename[-200:]

            sge_filename = folder_to_store_sge_files + sge_filename
            sge_file_list.append(sge_filename)

            content_for_sge_files = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M wei.herbert.chen@gmail.com         # email address

#$ -q all.q               # queue name
#$ -l h_rt=%s       # run time (hh:mm:ss)

%s
####$ -l hostname=compute-0-3

%s

echo "This job is DONE!"

exit 0
''' % (CONFIG_19, gpu_option_string, item)
            with open(sge_filename, 'w') as f_out:
                f_out.write(content_for_sge_files)
                f_out.write("\n")
        return sge_file_list

    @staticmethod
    def get_num_of_running_jobs():
        output = subprocess.check_output(['qstat'])
        all_entries = output.strip().split('\n')[2:]   # remove header
        all_entries = [item for item in all_entries if (not item.strip().split()[4] == 'dr')]   # remove job in "dr" state
        num_of_running_jobs = len(all_entries)
        # print('checking number of running jobs = %d\n' % num_of_running_jobs)
        return num_of_running_jobs

    @staticmethod
    def submit_sge_jobs_and_archive_files(job_file_lists,
                                          num,  # num is the max number of jobs submitted each time
                                          flag_of_whether_to_record_qsub_commands = False
                                          ):
        dir_to_archive_files = '../sge_files/archive/'

        if not os.path.exists(dir_to_archive_files):
            os.makedirs(dir_to_archive_files)

        assert(os.path.exists(dir_to_archive_files))
        sge_job_id_list = []
        for item in job_file_lists[0:num]:
            output_info = subprocess.check_output(['qsub', item]).strip()
            sge_job_id_list.append(output_info.split(' ')[2])
            print('submitting ' + str(item))
            subprocess.check_output(['mv', item, dir_to_archive_files]) # archive files
        return sge_job_id_list

    @staticmethod
    def submit_a_single_job_and_wait_until_it_finishes(job_sge_file):
        job_id = cluster_management.submit_sge_jobs_and_archive_files([job_sge_file], num=1)[0]
        while cluster_management.is_job_running_on_cluster(job_id):
            time.sleep(10)
        print "job (id = %s) done!" % job_id
        return job_id

    @staticmethod
    def run_a_command_and_wait_on_cluster(command):
        print 'running %s on cluster' % command
        sge_file = cluster_management.create_sge_files_for_commands([command])[0]
        id = cluster_management.submit_a_single_job_and_wait_until_it_finishes(sge_file)
        return id

    @staticmethod
    def get_output_and_err_with_job_id(job_id):
        temp_file_list = subprocess.check_output(['ls']).strip().split('\n')
        out_file = list(filter(lambda x: '.sge.o' + job_id in x, temp_file_list))[0]
        err_file = list(filter(lambda x: '.sge.e' + job_id in x, temp_file_list))[0]
        return out_file, err_file

    @staticmethod
    def get_sge_files_list():
        result = filter(lambda x: x[-3:] == "sge",subprocess.check_output(['ls', '../sge_files']).split('\n'))
        result = map(lambda x: '../sge_files/' + x, result)
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
                                               ):
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
            cluster_management.get_sge_dot_e_files_in_current_folder_and_handle_jobs_not_finished_successfully()
            try:
                temp_submitted_job_id = cluster_management.submit_new_jobs_if_there_are_too_few_jobs(num)
                submitted_job_id += temp_submitted_job_id
                submitted_job_id = list(filter(lambda x: cluster_management.is_job_running_on_cluster(x),
                                               submitted_job_id))   # remove finished id of finished jobs
                print "submitted_job_id = %s" % str(submitted_job_id)
                num_of_unsubmitted_jobs = len(cluster_management.get_sge_files_list())
            except:
                print("not able to submit jobs!\n")

        # then check if all jobs are done (not really, since there could be multiple cluster_management running)
        while cluster_management.get_num_of_running_jobs() > num_of_running_jobs_when_allowed_to_stop:
            time.sleep(10)
        return

    @staticmethod
    def is_job_running_on_cluster(job_sgefile_name):
        output = subprocess.check_output(['qstat', '-r'])
        return job_sgefile_name in output

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
        if cluster_management.is_job_running_on_cluster(job_sgefile_name):
            return 3  # not finished
        else:
            all_files_in_this_dir = subprocess.check_output(['ls']).strip().split()

            out_file_list = filter(lambda x: job_sgefile_name + ".o" in x, all_files_in_this_dir)
            err_file_list = filter(lambda x: job_sgefile_name + ".e" in x, all_files_in_this_dir)

            if len(out_file_list) == 0 or len(err_file_list) == 0:
                print "%s does not exist" % job_sgefile_name
                return -1  

            if latest_version:   # check output/error information for the latest version, since a job could be submitted multiple times
                job_serial_number_list = map(lambda x: int(x.split('.sge.o')[1]), out_file_list)
                job_serial_number_of_latest_version = max(job_serial_number_list)
                latest_out_file = filter(lambda x: str(job_serial_number_of_latest_version) in x, out_file_list)[0]
                latest_err_file = filter(lambda x: str(job_serial_number_of_latest_version) in x, err_file_list)[0]
                with open(latest_out_file, 'r') as out_f:
                    out_content = [item.strip() for item in out_f.readlines()]

                with open(latest_err_file, 'r') as err_f:
                    err_content = [item.strip() for item in err_f.readlines()]
                    err_content = filter(lambda x: x[:4] != 'bash', err_content)  # ignore error info starting with "bash"
                    err_content = filter(lambda x: not 'Using Theano backend' in x, err_content)
                    err_content = filter(lambda x: x != "", err_content)

                if (job_finished_message in out_content) and (len(err_content) != 0):
                    print "%s ends with exception" % job_sgefile_name
                    return 1  
                elif not job_finished_message in out_content:
                    print "%s aborted due to time limit or other reason" % job_sgefile_name
                    return 2  
                else:
                    print "%s finishes successfully" % job_sgefile_name
                    return 0  
            else:
                # TODO: handle this case
                return

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
                    print "restore sge_file: %s" % item
                    subprocess.check_output(['cp', dir_to_archive_files + item, folder_to_store_sge_files])
                    assert (os.path.exists(folder_to_store_sge_files + item))
                else:
                    print "%s not exists in %s" % (item, dir_to_archive_files)
            
            if status_code in (0, 1, 2):  # archive .o/.e files for finished jobs
                print "archive .o/.e files for %s" % item
                all_files_in_this_dir = subprocess.check_output(['ls']).strip().split()
                temp_dot_o_e_files_for_this_item = filter(lambda x: (item + '.o' in x) or (item + '.e' in x), 
                                                          all_files_in_this_dir)
                for temp_item_o_e_file in temp_dot_o_e_files_for_this_item:
                    subprocess.check_output(['mv', temp_item_o_e_file, dir_to_archive_files])
        return

    @staticmethod
    def get_sge_dot_e_files_in_current_folder_and_handle_jobs_not_finished_successfully(latest_version=True):
        all_files_in_this_dir = subprocess.check_output(['ls']).strip().split()
        sge_e_files = filter(lambda x: '.sge.e' in x, all_files_in_this_dir)
        sge_files = [item.split('.sge')[0] + '.sge' for item in sge_e_files]
        sge_files = list(set(sge_files))
        # print "sge_files = %s" % str(sge_files)
        cluster_management.handle_jobs_not_finished_successfully_and_archive(sge_files, latest_version)
        return
