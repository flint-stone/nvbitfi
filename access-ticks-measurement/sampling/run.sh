#!/bin/bash
set -x


WG=2048
WI=1
rm out-wg$WG-wi$WI.log

# SIZE=32
# SIZE=32
# while true 
# do 
#     # ./kernel $MEMORY_SIZE $WORKGROUP_SIZE $NUM_ACCESS $GPU_ID $WORKITEM_SIZE
#     ./kernel $SIZE $WG 62914560 0 $WI >> out-wg$WG-wi$WI.log
#     ((SIZE*=4))
#     if [[ $SIZE -gt 536870912 ]]
#     then
#         break
#     fi
# done



# # SIZE=32

PEER=1
WG=1
WI=16

for WI in 32 64 128 256 # 16 32 64 128 256
do
    SIZE=32
    rm p2p-log/out-p2p-peer$PEER-wg$WG-wi$WI.log
    while true 
    do 
        # ./kernel $SIZE 1 1000 0 >> out.log
        # 134217728/ 4 * 5 * 10
        # ./kernel-p2p $SIZE 1 1677721600 0 >> out.log
        ./kernel-p2p $SIZE $WG 1677721600 0 $PEER $WI 2>&1 >> p2p-log/out-p2p-peer$PEER-wg$WG-wi$WI.log
        #./kernel-p2p $SIZE 1 62914560 0 $PEER 2>&1 >> out-p2p-peer$PEER.log
        ((SIZE*=4))
        if [[ $SIZE -gt 536870912 ]]
        then
            break
        fi
    done
done

