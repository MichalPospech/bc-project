#!/bin/sh

PC=$1
PROJECT_DIR="/afs/ms/u/p/pospechmi/bc/bc-project"
COMMAND="cd $PROJECT_DIR &&  nohup mpiexec -n 4 $PROJECT_DIR/.env/bin/python -m mpi4py $PROJECT_DIR/train.py -f"
EXP=$2

for CONFIG in $PROJECT_DIR/configs/$EXP.*.json ; do
    echo "Running experiment for config $CONFIG on u2-$PC"
    ssh "u1-$PC" "$COMMAND $CONFIG 2>&1 > $PROJECT_DIR/log/$EXP.$PC.log" &
    PC=$((PC+1))
done
