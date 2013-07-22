#!/bin/bash

### Job settings
#SBATCH --job-name Mfftw
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16

### Environment setup
. /etc/profile
module load cuda/5.0

### Run tasks
srun cat /proc/cpuinfo > cpuinfo.dat

for i in 1 2 4 8 16
do
	echo
	echo "=== FFTW multiple threads ==="
	export OMP_NUM_THREADS=$i
	time srun -n1 simple_fftw_threads
	echo
done

wait
