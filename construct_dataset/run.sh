#!/bin/bash

asset_name="WithoutRuleLabelConsistencyCheck_FixedErrorInCounters_RemovedLangCheck_RemovedLengthCheck"
file_no_start=206
for i in {0..39}
do  
    file_no=$(expr $file_no_start + $i)
    screen -S "corona_tweets"$file_no -d -m taskset --cpu-list $i bash config.sh $asset_name $file_no
done

# asset_name="tweets_file"
# screen -S "screen"$asset_name -d -m bash config.sh $asset_name
# # bash config.sh $asset_name $start $stop
