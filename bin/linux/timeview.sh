#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=${DIR}/../..

pushd "${ROOT}" > /dev/null
source ${ROOT}/miniconda/bin/activate timeview $@
${ROOT}/miniconda/envs/timeview/bin/python -O -m timeview $@
popd > /dev/null
