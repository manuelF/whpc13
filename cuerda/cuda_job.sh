#!/bin/bash

### Las líneas #SBATCH configuran los recursos de la tarea
### (aunque parezcan estar comentadas)

### Nombre de la tarea
#SBATCH --job-name=nombre

### Cantidad de nodos a usar
#SBATCH --nodes=1

### GPUs por nodo (<= 2)
### OJO: todos los procesos del nodo ven todas las GPU
#SBATCH --gres=gpu:1

### Procesos por nodo
#SBATCH --ntasks-per-node=1

### Cores por proceso (OpenMP/Pthreads/etc)
#SBATCH --cpus-per-task=1


### Environment setup
. /etc/profile

### Environment modules
module load cuda/5.0

### Ejecutar la tarea
### NOTA: srun configura MVAPICH2 y MPICH con lo puesto arriba,
###       no hay que llamar a mpirun.

srun ./deviceQuery > gpuinfo.dat
srun hostname
echo 
echo "=== RESULTADOS ==="
srun ./qew_CUDA
echo "=================="
