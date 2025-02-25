#!/bin/bash
# MLO 2025, 🐯

mamba activate MLO


for GRIDSIZE in 48 64; do

(   
    echo $GRIDSIZE
    for J in $(LC_ALL=C seq 0.435 0.001 0.455); do
    (
            echo "J: $J"
            python metropolis.py -g $GRIDSIZE -b 0.0 -j $J --eq 100000 --prod 50000 --file binder_cum
            name="${GRIDSIZE}x${GRIDSIZE}"
            python analysis.py -g $GRIDSIZE -f binder_cum_$name
        
    )
    done

    wait

)
done