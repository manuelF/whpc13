#!/bin/bash

### Job settings
#SBATCH --job-name cuerOMP
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16

### Environment setup
. /etc/profile
module load cuda/5.0

### Run tasks
#srun cat /proc/cpuinfo > cpuinfo.dat

srun hostname

echo "== OMP performance" 
for i in 1 2 4 8 16
do
	export OMP_NUM_THREADS=$i
	time srun ./qew_OMP
	echo "=="
done

date

echo

wait
