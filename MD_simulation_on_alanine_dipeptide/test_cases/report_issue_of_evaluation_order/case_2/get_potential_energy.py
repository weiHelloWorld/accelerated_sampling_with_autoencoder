# get potential energy from biased_report file

import sys, os

temp = os.listdir('.')
filenames = filter(lambda x: x[0:13] == "biased_report", temp)

for report_file in filenames:
	print ('get energy file of ' + report_file)
	output_energy_file = report_file[0:7] + 'energy_' + report_file[14:]

	with open(report_file, 'r') as f_in:
		with open(output_energy_file, 'w') as f_out:
			for line in f_in:
				fields = line.strip().split(',')
				if(fields[0][0] != '#'):
					f_out.write(fields[2])
					f_out.write("\n")
					