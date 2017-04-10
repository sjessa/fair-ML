# Leverage points for fair machine learning

For our final project for COMP 551: Applied Machine Learning at McGill University in Winter 2017, we complete a survey of measures of discrimination in machine learning and methods for fair machine learning.

## Authors:  
Selin Jessa  
Elsa Riachi  
Adam Cavatassi  

## Run instructions

### Download and do minimal tidying of the data
```
cd analysis  
bash 01-download_data.sh
Rscript 02-prepare_data.R  
```

This downloads the Communities & Crime and Census datasets to a `data` directory, and the tidies up data for training learners. The results are `data/census/census_clean.tsv` where `sex` is the sensitive attribute and `income` the class, and `data/crime/crime_clean.tsv`, where `black` is the sensitive attribute and `crime` the class

### Train baseline classifiers on the Crime dataset
```
cd baseline  
python baseline.py
```

This trains three out-of-the-box classifiers: a Gaussian Naive Bayes classifier, a linear SVM, and a logistic regression classifier for the prediction task on the Crime dataset using 10-fold CV. The outputs are created in `baseline/output` and include the prediction files with class probabilities, predictions, and the known labels, as well as files to plot ROCs from mean TPR/FPR and a file with AUC scores.