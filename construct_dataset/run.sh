#!/bin/bash

asset_name="without_rule_label_consistency_check"
for i in {57..96}
do  
    # # # process_no=$(expr $i / 9)
    process_no=$(expr $i - 1 - 56)
    screen -S "screen"$i -d -m taskset --cpu-list $process_no bash config.sh $asset_name $i
    # bash config.sh $asset_name $i
    # # # sbatch config.sh $asset_name $i
    # echo $process_no
done

# asset_name="without_rule_label_consistency_check"
# process_no=1
# start=1
# stop=1
# screen -S "screen"$process_no -d -m taskset --cpu-list $process_no bash config.sh $asset_name $start $stop
# # bash config.sh $asset_name $start $stop