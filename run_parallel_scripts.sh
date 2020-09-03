#!/bin/bash

gcc -o run project_3.c 
gcc -o run1 project_3_omp_cg.c -fopenmp
gcc -o run2 project_3_omp_icg.c -fopenmp
gcc -o run3 project_3_omp_fg_vectorized.c -fopenmp
gcc -o run4 project_3_omp_fg.c -fopenmp

for N in 1 100 1000 10000
#for N in 10000
do
   for L in 10 100 1000
   #for L in 100 1000
   do
      for T in 2 4 6 8 10 12 14 16 20 25 30 35 40
      #for T in 2 4 6 8 10 12 14 16 20
      do

         echo "coarse-grained parallel code n,m="$N "l="$L "threads="$T
         ./run1 $N $N $L $T
         echo "medium grained parallel code n,m="$N "l="$L "threads="$T
         ./run2 $N $N $L $T
         echo "vectorized fine-grained parallel code n,m="$N "l="$L "threads="$T
         ./run3 $N $N $L $T
         echo "fine-grained parallel code n,m="$N "l="$L "threads="$T
         ./run4 $N $N $L $T
          

      done
   done
done
   
for N in 1 100 1000 10000
#for N in 10000
do
   for L in 10 100 1000
   #for L in 100 1000
   do
      echo "seq code n,m="$N "l="$L
      ./run $N $N $L
   done
done

   







