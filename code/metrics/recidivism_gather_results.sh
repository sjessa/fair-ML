#!/bin/bash


echo "gathering in process..... please wait"

echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > recidivism_comparison.csv
echo -n Raw Data >> recidivism_comparison.csv 
python metrics.py Non-White Recidivism recidivism_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv

find ../output -maxdepth 1 -name "*recidivism_massaged*" -print >> predictions_list.txt
find ../output -maxdepth 1 -name "*recidivism_reweighed*" -print >> predictions_list.txt
find ../output -maxdepth 1 -name "*recidivism*" -name "*pred*" ! -name "*lock*" -print >> predictions_list.txt

while read prediction
do
    echo -n "$prediction" >> recidivism_comparison.csv
    if [[ ${prediction} != *"pred"* ]];then
	python metrics.py non_white recidivism "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv
    else
   	python metrics.py non_white pred "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv
    fi
done < predictions_list.txt


echo ,ACCURACY:,PRECISION:,RECALL:,F1 SCORE:, > recidivism_accuracy.csv

rm predictions_list.txt
find ../output -maxdepth 1 -name "*recidivism*" -name "*pred*" ! -name "*lock*" -print > predictions_list.txt

while read prediction2
do
    echo -n "$prediction2", >> recidivism_accuracy.csv
    python accuracy.py pred class "$prediction2" | cut -d' ' -f2 | paste -sd ',' >> recidivism_accuracy.csv
  
done < predictions_list.txt

echo "all done!"
