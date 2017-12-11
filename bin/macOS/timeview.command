#!/usr/bin/env bash

echo "Due to a bad interaction between Qt and macOS-Sierra, please press CMD-TAB twice after the TimeView window opens."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=${DIR}/../..

pushd "${ROOT}" > /dev/null

source "${ROOT}/miniconda/bin/activate" timeview  # required for R to work (and probably a good idea)

ln -sf "${ROOT}/miniconda/envs/timeview/bin/python" "${ROOT}/bin/TimeView"  # hack for OSX global QMenu
"${ROOT}/bin/TimeView" -O -m timeview $@
rm "${ROOT}/bin/TimeView"

popd > /dev/null
