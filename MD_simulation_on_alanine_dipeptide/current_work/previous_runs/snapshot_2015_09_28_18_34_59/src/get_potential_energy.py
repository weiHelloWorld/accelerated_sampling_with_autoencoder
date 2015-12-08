# get potential energy from biased_report file

import sys, os

temp = os.listdir('../target')
filenames = filter(lambda x: x[0:13] == "biased_report" and x[-10:-4] != "energy", temp) # get only report files

for report_file in filenames:
	print ('get energy file of ' + report_file)
	output_energy_file = report_file[:-4] + '_energy.txt'

	report_file = '../target/' + report_file
	output_energy_file = '../target/' + output_energy_file

	with open(report_file, 'r') as f_in:
		with open(output_energy_file, 'w') as f_out:
			for line in f_in:
				fields = line.strip().split(',')
				if(fields[0][0] != '#'):
					f_out.write(fields[1])
					f_out.write("\n")
					