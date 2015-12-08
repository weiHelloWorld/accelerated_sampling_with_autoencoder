from numpy import arange
import sys


PC1_start = float(sys.argv[1])
PC1_end = float(sys.argv[2])
PC1_interval = float(sys.argv[3])

PC2_start = float(sys.argv[4])
PC2_end = float(sys.argv[5])
PC2_interval = float(sys.argv[6])


content = "["

for PC1 in arange(PC1_start, PC1_end, PC1_interval):
	for PC2 in arange(PC2_start, PC2_end, PC2_interval):
		content += "[%s, %s],"%(str(PC1), str(PC2))

content += "]"

print content