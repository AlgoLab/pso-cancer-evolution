#!/bin/bash

cd $1
number_of_files=$[$(ls | wc -l)/4]

cd ~/github/pso-cancer-evolution-2.0/
mkdir test_results
source ./env/bin/activate


for ((i = 1 ; i <= $number_of_files ; i++)); do
	python3 psosc.py -i "$1/sim_${i}_scs.txt" --truematrix "$1/sim_${i}_truescs.txt"
done


folder_list=$(find . -type d -printf '%f\n' | grep "results20*")

for folder in $folder_list; do
	mv ./$folder/result* ./test_results
	rm -r ./$folder
done
