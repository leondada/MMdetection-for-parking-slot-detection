#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    $(dirname "$0")/train.py  configs/FedPSD/yolo.py --launcher pytorch ${@:3}
