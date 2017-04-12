#!/bin/bash

# Download Communities & Crime dataset
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data \
	--directory-prefix=../data/crime/ -nc

# Metadata
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names \
	--directory-prefix=../data/crime/ -nc

( echo "variable,type" & \
    grep '^@attribute' ../data/crime/communities.names \
    | sed 's/\ /,/g' \
    | cut -d',' -f2,3 \
    ) > ../data/crime/communities_metadata.csv

# Download Census Income dataset
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data \
	--directory-prefix=../data/census/ -nc

# Test set
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test \
	--directory-prefix=../data/census/ -nc

# Metadata
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names \
	--directory-prefix=../data/census/ -nc

tail -n 14 ../data/census/adult.names | cut -d':' -f 1 > ../data/census/census_header.txt