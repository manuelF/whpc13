#!/bin/bash
### Job settings
#SBATCH --job-name fft1
#SBATCH --nodes 1
#SBATCH --ntasks 1

### Environment setup
. /etc/profile
module load cuda/5.0

### Run tasks

echo "=== FFTW single thread ==="
time srun ./simple_fftw
echo

wait
