#!/bin/bash

prefix=$1

current_time=$(date -u +%Y%m%d%H%M%S)
dir_name=${prefix}"/ss_"${current_time}

mkdir -p ${dir_name}

for item in README.md resources src target; do
	echo "copying "${item}
	rsync -ar --exclude='.*' ${item} ${dir_name}
done
