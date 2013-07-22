#!/bin/bash
### Job settings
#SBATCH --job-name cuffthrust
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1

### Environment setup
. /etc/profile
module load cuda/5.0

### Run tasks
srun cat /proc/cpuinfo > cpuinfo.dat
srun deviceQuery > gpuinfo.dat

echo
echo "=== CUFFT ==="
time srun --gres=gpu:1 -n1 simple_cufft
echo 

wait
