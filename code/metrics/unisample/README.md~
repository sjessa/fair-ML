# Run instructions for metrics.py

```
python metrics.py protected output prediction_file.tsv [False]
```

This script will print the Impact Ratio, Elift Ratio, and Odds Ratio according to the definitions our literature. The protected characteristic is user defined, as is the discriminated output. `protected` is the protected feature vector column in `prediction_file.tsv`, and `output` is the prediction vector column. False is an optional argument that can be used if the desireable outcome of the dataset is 0 rather than 1. Note that this script is only valid when `protected` and `output` are binary vectors. The prediction file can be a .tsv or a .csv.


# instructions for gathering results 

To collect the results for the crime dataset from all the learners thus far, simply run the `crime_gathering_results.sh` script. 

```
./crime_gathering_results.sh
```

The discrimination metrics for each of the classifiers will be written in `crime_comparison.csv`, and the accuracy metrics for each of the predictions will be written in `crime_accuracy.csv`.

To collect the results for the recidivism dataset from all the learners thus far, simply run the `recidivism_gathering_results.sh` script. 

```
./recidivism_gathering_results.sh
```

The discrimination metrics for each of the classifiers will be written in `recidivism_comparison.csv`, and the accuracy metrics for each of the predictions will be written in `recidivism_accuracy.csv`.
