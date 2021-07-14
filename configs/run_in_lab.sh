#!/bin/sh

PC=$2
COMMAND=$1

for CONFIG in ${@:3}; do
    echo "Running experiment for config $CONFIG"
    ssh "u2-$PC" $COMMAND $CONFIG    
done