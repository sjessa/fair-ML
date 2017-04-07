#!/bin/bash

# Download UMI Communities & Crime dataset
wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data \
	--directory-prefix=data/

wget \
	https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names \
	--directory-prefix=data/