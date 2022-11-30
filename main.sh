#!/usr/bin/env bash
src="rak-communities-openmp"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -c unlimited
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
g++ -std=c++17 -O3 -fopenmp -g -rdynamic main.cxx
stdbuf --output=L ./a.out ~/data/GAP-road.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/it-2004.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/webbase-2001.mtx 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-twitter.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-web.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/sk-2005.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-kron.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-urand.mtx    2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/AGATHA_2015.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/MOLIERE_2016.mtx 2>&1 | tee -a "$out"
