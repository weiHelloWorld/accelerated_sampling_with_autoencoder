#!/bin/bash

if [[ "$#" -eq 1 ]]; then
	prefix=$1
else
	prefix="../../previous_runs"
fi

current_time=$(date -u +%Y_%m_%d_%H_%M_%S)
dir_name=${prefix}"/snapshot_"${current_time}

mkdir -p ${dir_name}

for item in README.md resources src target; do
	echo "copying "${item}
	rsync -ar --exclude='.*' ${item} ${dir_name}
done
