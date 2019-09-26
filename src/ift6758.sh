#!/bin/bash
# Initialize parameters specified from command line

usage() { echo "Usage: $0 -i <test_data> -o <results> " 1>&2; exit 1; }

declare test_data=""
declare results=""

while getopts ":i:o:" arg; do
	case "${arg}" in
		i)
			test_data=${OPTARG}
			;;
		o)
			results=${OPTARG}
			;;
	esac
done

echo $test_data

echo $results

# shellcheck disable=SC2162
read -p "Press [Enter] key to start backup..."

# shellcheck disable=SC2086
python __init__.py $test_data $results
