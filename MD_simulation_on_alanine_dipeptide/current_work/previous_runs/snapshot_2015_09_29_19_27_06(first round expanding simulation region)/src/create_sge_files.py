#!/usr/bin/python

list_of_force_constant = ["1000"]
list_of_potential_center = [["-0.3", "-0.5"], ["-0.3", "-0.4"], ["-0.3", "-0.3"], ["-0.2", "-0.5"], ["-0.2", "-0.4"], ["-0.2", "-0.3"], ["-0.6", "-0.2"], ["-0.5", "-0.2"], ["-0.4", "-0.2"], ["-0.3", "-0.2"], ["-0.2", "-0.2"], ["-0.1", "-0.2"], ["0", "-0.2"], ["0.1", "-0.2"], ["0.2", "-0.2"]]
command_on_google = ""

for potential_center in list_of_potential_center:

	parameter_list = ("50", "10000", "1000", potential_center[0], potential_center[1])
	
	file_name = "../sge_files/job_biased_%s_%s_%s_%s_%s.sge" % parameter_list
	command = "python ../src/biased_simulation.py %s %s %s %s %s" % parameter_list

	print("creating %s" % file_name)

	content_for_sge_files = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment 
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M weichen9@illinois.edu         # email address

#$ -q all.q               # queue name
#$ -l h_rt=50:00:00       # run time (hh:mm:ss)


echo " "
echo "-------------------"
echo "This is a $ENVIRONMENT job"
echo "This job was submitted to the queue: $QUEUE"
echo "The job's id is: $JOB_ID"
echo "The job's name is: $JOB_NAME"
echo "The job's home dir is: $SGE_O_HOME"
echo "The job's working dir is: $SGE_O_WORKDIR"
echo "The host node of this job is: $SGE_O_HOST"
echo "The master node of this job is: $HOSTNAME"
echo "The number of cores used by this job: $NSLOTS"
echo "This job was submitted by: $SGE_O_LOGNAME"
echo "-------------------"
echo Running on host `hostname`
echo Time is `date`
echo "-------------------"
echo " "

%s

echo " "
echo "This job is DONE!"
echo " "

exit 0
''' % command

	with open(file_name, 'w') as f_out:
		f_out.write(content_for_sge_files);
		f_out.write("\n")

	# additional: generate file to run on Google Compute Engine

	command_on_google = command_on_google + 'sudo ' + command + "&\n"

file_of_running_on_google = "../sge_files/running_on_google.sh"
with open(file_of_running_on_google, 'w') as f_out:
	f_out.write(command_on_google)
	f_out.write("\n")
	