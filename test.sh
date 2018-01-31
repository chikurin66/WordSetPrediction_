#!/bin/sh

# 評価


for i in 0 1
do
if [ $i = 0 ]
then
    slan=en
    tlan=ja
    echo "iii"
fi
if [ $i = 1 ]
then
    slan=ja
    tlan=en
    echo "iiii"
fi
echo $i
echo $slan $tlan
done
