#!/bin/bash
# MLO 2025, 🐯

mamba activate MLO


for GRIDSIZE in 32; do

(   
    echo $GRIDSIZE
    for J in $(LC_ALL=C seq 0.25 0.01 0.75); do
    (
            echo "J: $J"
            python metropolis.py -g $GRIDSIZE -b 0.0 -j $J --eq 100000 --prod 50000
            name="${GRIDSIZE}x${GRIDSIZE}"
            # python analysis.py -g $GRIDSIZE -f output_$name/
        
    ) &
    done

    wait

)
done