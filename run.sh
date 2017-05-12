#!/bin/bash
min_n=100
add_n=100
max_n_float=9000
max_n_double=4500

n=$min_n
while [ $n -le $max_n_float ]
do
	echo "float... $n"
	./test_float.exe -n $n 1>>1.log 2>>2.log
	n=$(( n+add_n ))
done

n=$min_n
while [ $n -le $max_n_double ]
do
	echo "double... $n"
	./test_double.exe -n $n 1>>1.log 2>>2.log
	n=$(( n+add_n ))
done