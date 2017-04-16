#!/bin/bash


echo "gathering in process..... please wait"

echo ,Sex: > recidivism_comparison.csv
echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, >> recidivism_comparison.csv
echo -n Raw Data, >> recidivism_comparison.csv 
python metrics.py Sex Recidivism recidivism_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv
echo >> recidivism_comparison.csv

echo ,Hispanic: >> recidivism_comparison.csv
echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, >> recidivism_comparison.csv
echo -n Raw Data, >> recidivism_comparison.csv 
python metrics.py Hispanic Recidivism recidivism_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv
echo >> recidivism_comparison.csv

echo ,Non-White: >> recidivism_comparison.csv
echo ,IMPACT RATIO:,ELIFT RATIO:,ODDS RATIO:, >> recidivism_comparison.csv
echo -n Raw Data, >> recidivism_comparison.csv 
python metrics.py Non-White Recidivism recidivism_clean.csv False | cut -d' ' -f3 | paste -sd ',' >> recidivism_comparison.csv
echo >> recidivism_comparison.csv


echo "all done!"
