#!/bin/bash

echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > comparison.csv
echo -n Raw Data, >> comparison.csv 
python metrics.py black crime crime_clean.tsv False | cut -d' ' -f3 | paste -sd ',' >> comparison.csv

find ../output -maxdepth 1 -name "*crime*" -print > predictions_list.txt
find ../output -maxdepth 1 -name "*pred*" -print >> predictions_list.txt

while read prediction
do
    echo -n "$prediction", >> comparison.csv
    python metrics.py black pred "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> comparison.csv
done < predictions_list.txt
