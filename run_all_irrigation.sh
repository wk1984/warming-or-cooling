#!/bin/bash
#
#
# Check ats version ...

ats --version

arr0=( "02peat" "05peat" )
arr1=( "cd" "cw" "wd" "ww")

for i1 in "${arr1[@]}"
do
for i0 in "${arr0[@]}"
do
cd /Users/kangwang/Documents/GitHub/warming-or-cooling
cd regions/$i1/$i0/i
echo regions/$i1/$i0/i
./run.sh &
done
done

sleep 5