#!/bin/bash


echo "rounding in process..... please wait"
find ./ -maxdepth 1 -name "*.csv*" ! -name "*recidivism*" -print > rounding_list.txt

while read round
do
    python round.py "$round"
   
done < rounding_list.txt

echo "all done!"
