# Leverage points for fair machine learning

For our final project for COMP 551: Applied Machine Learning at McGill University in Winter 2017, we complete a survey of measures of discrimination in machine learning and methods for fair machine learning.

## Authors:  
Selin Jessa  
Elsa Riachi  
Adam Cavatassi  

## Run instructions
From within `analysis`:  
1. Run `bash 01-download_data.sh` to download the Communities & Crime and Census datasets to a `data` directory
2. Run `Rscript 02-prepare_data.R` to tidy up data for training learners.
This creates `data/census/census_clean.tsv` where `black` is the sensitive attribute and `crime` the class, and `data/crime/crime_clean.tsv` where `sex` is the sensitive attribute and `income` is the class.
