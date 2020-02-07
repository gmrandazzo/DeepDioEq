#!/usr/bin/env bash
epochs=1000

function bfsearch(){
    units=(8
           16
           32
           64
           128
           256
           512
    )
    
    layers=(1 2 3)
    
    batch_size=(10 100 200)
    
    for b in ${batch_size[*]}; do 
        for u in ${units[*]}; do
            for l in ${layers[*]}; do
                python3 train.py ${epochs} ${b} ${u} ${l}
            done
        done
    done
    
}

bfsearch
