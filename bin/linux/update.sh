#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=${DIR}/../..

"${ROOT}/miniconda/envs/timeview/bin/git" pull

./install.sh  # because packages may have changed
