#!/bin/bash

set -e -x

CLU_DST="${CLU_DST:-/tmp/clu}"
CLU_ENV="${CLU_ENV:-/tmp/clu_env}"

copybara third_party/py/clu/copy.bara.sky local .. \
    --folder-dir="${CLU_DST}" --ignore-noop

# Note: we're reusing the environment if it already exists.
mkdir -p "${CLU_ENV}"
cd "${CLU_ENV}"
python3 -m virtualenv .
. bin/activate

cd "${CLU_DST}"
pip install . .[test]

pytest
