#!/bin/bash
unset XDG_RUNTIME_DIR
/home/zhenan/projects/def-mpf/zhenan/julia/conda/3/bin/jupyter notebook --ip $(hostname -f) --no-browser