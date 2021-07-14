#!/bin/sh

SEEDS=${@:2}
CONFIG_NAME=$1

for SEED in $SEEDS; do
    cp $CONFIG_NAME.template.json $CONFIG_NAME.$SEED.json
    sed -i "s/SEED/$SEED/g" $CONFIG_NAME.$SEED.json
done