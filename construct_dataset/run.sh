#!/bin/bash

# start_init=1
# process_no_init=1
# for i in {0..55} # no of CPUs
# do  
#     process_no=$(($process_no_init + $i))
#     start=$(($start_init + 5*i))
#     stop=$(($start + 4))
#     screen -S "screen"$process_no -d -m taskset --cpu-list $process_no python3 scripts/construct_covid19_dataset.py $process_no $start $stop
# done

asset_name="without_rule_label_consistency_check"
process_no=1
start=1
stop=1
# screen -S "screen"$process_no -d -m taskset --cpu-list $process_no bash config.sh $asset_name $start $stop
bash config.sh $asset_name $start $stop