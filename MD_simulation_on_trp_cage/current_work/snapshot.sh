#!/bin/bash

current_time=$(date -u +%Y_%m_%d_%H_%M_%S)
dir_name="../../previous_runs/snapshot_"${current_time}

mkdir -p ${dir_name}

for item in README.md resources src target tests; do
	echo "copying "${item}
	rsync -ar --exclude='.*' ${item} ${dir_name}
done
