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

export LARGO=1000000
for i in 1 2 4 8 16
do
	export OMP_NUM_THREADS=$i
	echo "resultados para $OMP_NUM_THREADS OMP-threads "
	srun ./omp.out $LARGO
done

wait
