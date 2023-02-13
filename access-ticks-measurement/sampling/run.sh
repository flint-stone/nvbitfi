#!/bin/bash
# set -x
rm out.log

SIZE=32
while true 
do 
    # ./kernel $SIZE 1 1000 0 >> out.log
    # 134217728/ 4 * 5 * 10
    ./kernel $SIZE 1 1677721600 0 >> out.log
    ((SIZE*=4))
    if [[ $SIZE -gt 536870912 ]]
    then
        break
    fi
done
