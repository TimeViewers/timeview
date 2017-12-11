#!/usr/bin/env bash

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#ROOT=${DIR}/..
#source ${ROOT}/miniconda/envs/timeview/bin/activate timeview

# http://www.sphinx-doc.org/en/stable/builders.html
make qthelp

# http://doc.qt.io/qt-5/qthelp-framework.html
qcollectiongenerator build/qthelp/TimeView.qhcp

mv build/qthelp/TimeView.qhc build/qthelp/TimeView.qch ../timeview/gui
