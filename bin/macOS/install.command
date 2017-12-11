#!/usr/bin/env bash

URL=https://repo.continuum.io/miniconda/
FILE=Miniconda3-latest-MacOSX-x86_64.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=${DIR}/../..

cd "${ROOT}"
if [ ! -f ${ROOT}/miniconda/bin/python ]; then
	echo "Downloading anaconda data science platform"
	curl ${URL}${FILE} -o miniconda.sh
	echo "Installing anaconda"
	bash miniconda.sh -b -p "${ROOT}/miniconda"
	rm miniconda.sh
fi

echo "Creating environment, please wait"
PATH="${ROOT}/miniconda/bin:${PATH}"
conda env create --force -f environment.yml | awk -v n=8 'NR>n{print line[NR%n]};{line[NR%n]=$0}'  # skip activate message
conda clean -y -a
