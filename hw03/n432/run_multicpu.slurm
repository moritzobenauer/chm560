#!/bin/bash
#MLO 2025 @ Princeton University

#SBATCH --job-name=lammpst1      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=1:59:59          # total run time limit (HH:MM:SS)
#SBATCH --array=0-5

ARGS=(15061999 10052004 31081973 25021972 30071950 21071999)



SEED=${ARGS[$SLURM_ARRAY_TASK_ID]}

echo $SEED

STEPS_EQ=8000000
STEPS_PROD=10000000
N=432

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_d9_double_aocc -in lj_equil.lmp -var seed $SEED -var run_steps $STEPS_PROD -var eq_steps $STEPS_EQ -var particles $N

#srun $HOME/.local/bin/lmp_intel -sf intel -in lj_equil.lmp -var seed $SEED -var run_steps $STEPS_PROD -var eq_steps $STEPS_EQ -var particles $N
