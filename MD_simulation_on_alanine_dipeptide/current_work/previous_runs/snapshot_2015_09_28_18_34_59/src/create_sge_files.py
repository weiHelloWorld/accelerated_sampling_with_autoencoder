

for force_constant in ['1', '5', "10", "50", "100", "300", "500", "1000", "3000", "5000", "10000"]:
	file_name = "job_biased_%s.sge" % force_constant

	command = "python biased_simulation.py %s" % force_constant

	content = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment 
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M weichen9@illinois.edu         # email address

#$ -q all.q               # queue name
#$ -l h_rt=10:00:00       # run time (hh:mm:ss)


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
		f_out.write(content);
		f_out.write("\n")