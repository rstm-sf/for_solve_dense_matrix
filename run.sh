#!/bin/bash
max_n_float=9000
max_n_double=4500

echo "float, mklseq..."
n=100
while [ $n -le $max_n_float ]
do
	./test_float_mklseq.exe -n $n -t 2 1>>1.log 2>>2.log
	n=$(( n+100 ))
done

echo "float, mklpar..."
n=100
while [ $n -le $max_n_float ]
do
	./test_float_mklpar.exe -n $n -t 2 1>>1.log 2>>2.log
	n=$(( n+100 ))
done

echo "double, mklseq..."
n=100
while [ $n -le $max_n_double ]
do
	./test_double_mklseq.exe -n $n -t 2 1>>1.log 2>>2.log
	n=$(( n+100 ))
done

echo "double, mklpar..."
n=100
while [ $n -le $max_n_double ]
do
	./test_double_mklpar.exe -n $n -t 2 1>>1.log 2>>2.log
	n=$(( n+100 ))
done

echo "float, cudatoolkit..."
n=100
while [ $n -le $max_n_float ]
do
	./test_float_mklpar.exe -n $n -t 3 1>>1.log 2>>2.log
	n=$(( n+100 ))
done

echo "double, cudatoolkit..."
n=100
while [ $n -le $max_n_double ]
do
	./test_double_mklpar.exe -n $n -t 3 1>>1.log 2>>2.log
	n=$(( n+100 ))
done