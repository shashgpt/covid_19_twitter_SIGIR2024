#!/bin/bash

# for pid in {407788..411986}
# do
#     wait $pid
# done

asset_name="WithoutRuleLabelConsistencyCheck_FixedErrorInCounters_RemovedLangCheck_RemovedLengthCheck"
file_no_start=493
for i in {0..39}
do  
    file_no=$(expr $file_no_start + $i)
    screen -S "corona_tweets"$file_no -d -m taskset --cpu-list $i bash config.sh $asset_name $file_no
done
