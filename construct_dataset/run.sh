#!/bin/bash

asset_name="without_rule_label_consistency_check_fixed_error_in_counters"
file_no_start=153
for i in {0..39}
do  
    file_no=$(expr $file_no_start + $i)
    screen -S "corona_tweets"$file_no -d -m taskset --cpu-list $i bash config.sh $asset_name $file_no
done

# asset_name="without_rule_label_consistency_check"
# process_no=1
# start=1
# stop=1
# screen -S "screen"$process_no -d -m taskset --cpu-list $process_no bash config.sh $asset_name $start $stop
# # bash config.sh $asset_name $start $stop