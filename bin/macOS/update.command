#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=${DIR}/../..

cd "${DIR}"
"${ROOT}/miniconda/envs/timeview/bin/git" pull
"${DIR}/install.command"  # because packages may have changed
