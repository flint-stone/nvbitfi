#!/bin/bash
set -x


WG=1
WI=32
rm out-wg$WG-wi$WI.log

# SIZE=32
SIZE=32
while true 
do 
    # ./kernel $MEMORY_SIZE $WORKGROUP_SIZE $NUM_ACCESS $GPU_ID $WORKITEM_SIZE
    ./kernel $SIZE $WG 62914560 0 $WI >> out-wg$WG-wi$WI.log
    ((SIZE*=4))
    if [[ $SIZE -gt 536870912 ]]
    then
        break
    fi
done



# # SIZE=32
# SIZE=32
# PEER=3
# rm out-p2p-peer$PEER.log
# while true 
# do 
#     # ./kernel $SIZE 1 1000 0 >> out.log
#     # 134217728/ 4 * 5 * 10
#     # ./kernel-p2p $SIZE 1 1677721600 0 >> out.log
#     ./kernel-p2p $SIZE 1 1677721600 0 $PEER 2>&1 >> out-p2p-peer$PEER.log
#     #./kernel-p2p $SIZE 1 62914560 0 $PEER 2>&1 >> out-p2p-peer$PEER.log
#     ((SIZE*=4))
#     if [[ $SIZE -gt 536870912 ]]
#     then
#         break
#     fi
# done

