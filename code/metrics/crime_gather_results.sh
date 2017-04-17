#!/bin/bash


echo "gathering in process..... please wait"

echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > crime_comparison.csv
echo -n Raw Data, >> crime_comparison.csv 
python metrics.py black crime crime_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> crime_comparison.csv

find ../output -maxdepth 1 -name "*crime_*" ! -name "*recidivism*" -print > predictions_list.txt
find ../output -maxdepth 1 -name "*pred*" ! -name "*recidivism*" -print >> predictions_list.txt

while read prediction
do
    echo -n "$prediction", >> crime_comparison.csv
    if [[ ${prediction} != *"pred"* ]];then
	python metrics.py black crime "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> crime_comparison.csv
    else
   	python metrics.py black pred "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> crime_comparison.csv
    fi
done < predictions_list.txt

echo ,ACCURACY:,PRECISION:,RECALL:,F1 SCORE:, > crime_accuracy.csv

rm predictions_list.txt
find ../output -maxdepth 1 -name "*pred*" ! -name "*recidivism*" -print > predictions_list.txt

while read prediction2
do
    echo -n "$prediction2", >> crime_accuracy.csv
    python accuracy.py pred class "$prediction2" | cut -d' ' -f2 | paste -sd ',' >> crime_accuracy.csv
  
done < predictions_list.txt

echo "all done!"
