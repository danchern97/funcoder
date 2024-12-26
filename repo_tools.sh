#!/bin/bash

if [[ $1 == "clean" ]]; then
    rm -rfv ./.pytest_cache/
    rm -rfv ./build/
    rm -rfv ./dist/
    rm -rfv ./funcoder.egg-info/
    rm -rfv ./*/__pycache__/ ./*/*/__pycache__/ ./*/*/*/__pycache__/ ./*/*/*/*/__pycache__/
    echo "cache cleaned"
elif [[ $1 == "install" ]]; then
    conda create -y -n funcoder python=3.10
    conda activate funcoder
    python -m pip install --editable .
elif [[ $1 == "build" ]]; then
    python -m pip install --upgrade build
    python -m build
elif [[ $1 == "test" ]]; then
    python -m pytest .
elif [[ $1 == "publish" ]]; then
    python -m twine upload dist/*
else
    echo "invalid command"
fi
