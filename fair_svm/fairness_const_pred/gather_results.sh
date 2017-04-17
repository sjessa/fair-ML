#!/bin/bash


echo "gathering in process..... please wait"


# accuracy constrained
echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > fair_logr_acc_crime_comparison.csv
echo -n Raw Data >> fair_logr_acc_crime_comparison.csv 
python ../../code/metrics/metrics.py black crime ../../code/metrics/crime_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> fair_logr_acc_crime_comparison.csv

find ./ -maxdepth 1 -name "*acc*" -name "*.csv*" ! -name "*logr*" -print > predictions_list.txt
echo "$(sort predictions_list.txt)" > predictions_list.txt

while read prediction
do
    echo -n "$prediction" >> fair_logr_acc_crime_comparison.csv
       	python ../../code/metrics/metrics.py black prediction "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> fair_logr_acc_crime_comparison.csv
    
done < predictions_list.txt

echo ,ACCURACY:,PRECISION:,RECALL:,F1 SCORE:, > fair_logr_acc_crime_accuracy.csv

while read prediction2
do
    echo -n "$prediction2", >> fair_logr_acc_crime_accuracy.csv
    python ../../code/metrics/accuracy.py prediction actual "$prediction2" | cut -d' ' -f2 | paste -sd ',' >> fair_logr_acc_crime_accuracy.csv
  
done < predictions_list.txt

rm predictions_list.txt


# fairness constrained
echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, > fair_logr_fairness_crime_comparison.csv
echo -n Raw Data >> fair_logr_fairness_crime_comparison.csv 
python ../../code/metrics/metrics.py black crime ../../code/metrics/crime_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> fair_logr_fairness_crime_comparison.csv

find ./ -maxdepth 1 -name "*fairness*" -name "*.csv*" ! -name "*logr*" -print > predictions_list.txt
echo "$(sort predictions_list.txt)" > predictions_list.txt

while read prediction
do
    echo -n "$prediction" >> fair_logr_fairness_crime_comparison.csv
       	python ../../code/metrics/metrics.py black prediction "$prediction" False | cut -d' ' -f3 | paste -sd ',' >> fair_logr_fairness_crime_comparison.csv
    
done < predictions_list.txt

echo ,ACCURACY:,PRECISION:,RECALL:,F1 SCORE:, > fair_logr_fairness_crime_accuracy.csv

while read prediction2
do
    echo -n "$prediction2", >> fair_logr_fairness_crime_accuracy.csv
    python ../../code/metrics/accuracy.py prediction actual "$prediction2" | cut -d' ' -f2 | paste -sd ',' >> fair_logr_fairness_crime_accuracy.csv
  
done < predictions_list.txt
echo "all done!"
