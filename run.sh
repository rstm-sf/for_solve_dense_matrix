#!/bin/bash
n=2

# continue until $n equals 8192
while [ $n -le 8192 ]
do
	./test.exe -n $n 1>>1.log 2>>2.log
	n=$(( n*2 ))
done