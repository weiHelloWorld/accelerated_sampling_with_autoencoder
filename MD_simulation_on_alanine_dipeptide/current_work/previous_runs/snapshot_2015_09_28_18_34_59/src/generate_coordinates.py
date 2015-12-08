# this script is used to generate coordinates after processing for matlab
# the output consists of 21 columns

import sys, os

temp = os.listdir('../target')
filenames = filter(lambda x: x[-3:] == "pdb",temp)  # more general: generate coordinates for all pdb files

index_of_backbone_atoms = ['2', '5', '7', '9', '15', '17', '19']

for input_file in filenames:
    print ('generating coordinates of ' + input_file)
    output_file = input_file[:-4] + '_coordinates.txt'
    
    input_file = '../target/' + input_file
    output_file = '../target/' + output_file
    
    with open(input_file) as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                fields = line.strip().split()
                if (fields[0] == 'ATOM' and fields[1] in index_of_backbone_atoms):
                    f_out.write(reduce(lambda x,y: x + '\t' + y, fields[6:9]))
                    f_out.write('\t')
                elif fields[0] == "MODEL" and fields[1] != "1":
                    f_out.write('\n')
