'''
This program takes a "terminal command" (should be within quotation mark) as an argument, 
generate corresponding sge file, and qsub it.
'''

import sys, subprocess

whether_to_qsub = sys.argv[1]
command_in_sge_file = sys.argv[2]

content_for_sge_file = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment 
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M wei.herbert.chen@gmail.com         # email address

#$ -q all.q               # queue name
#$ -l h_rt=24:00:00       # run time (hh:mm:ss)
#$ -l hostname=compute-0-3

%s

echo "This job is DONE!"

exit 0
''' % command_in_sge_file

sge_filename = '../sge_files/' + command_in_sge_file.replace(' ', '_') + '.sge'

with open(sge_filename, 'w') as sge_file:
	sge_file.write(content_for_sge_file)

if whether_to_qsub == '1':
	subprocess.check_output(['qsub', sge_filename])
