#!/bin/bash


echo "gathering in process..... please wait"

echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > comparison.csv
echo -n Raw Data, >> comparison.csv 
python metrics.py black crime crime_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> comparison.csv

find ../output -maxdepth 1 -name "*crime*" -print > predictions_list.txt
find ../output -maxdepth 1 -name "*pred*" -print >> predictions_list.txt

while read prediction
do
    echo -n "$prediction", >> comparison.csv
    if [[ ${prediction} != *"pred"* ]];then
	python metrics.py black crime "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> comparison.csv
    else
   	python metrics.py black pred "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> comparison.csv
    fi
done < predictions_list.txt

echo ,ACCURACY:,PRECISION:,RECALL:,F1 SCORE:, > accuracy.csv

rm predictions_list.txt
find ../output -maxdepth 1 -name "*pred*" -print > predictions_list.txt

while read prediction2
do
    echo -n "$prediction2", >> accuracy.csv
    python accuracy.py pred class "$prediction2" | cut -d' ' -f2 | paste -sd ',' >> accuracy.csv
  
done < predictions_list.txt

echo "all done!"
