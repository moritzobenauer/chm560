# HW03

## slurm input for della9

```bash
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

```


## LAMMPS input script for NVT equilibration and NVE production

```
units         lj                   # Use reduced Lennard-Jones units
atom_style    atomic               # Simple atomic interactions

# ---------- Define Variables (set via command line) ----------
variable      seed index 12345     # Seed provided via command line
variable      run_steps index 100000  # Run steps provided via command line
variable      particles index 108 	
variable      eq_steps index 50000

variable      N equal ${particles}           # Number of particles
variable      rho_mass equal 0.80726    # Mass density in reduced LJ units
variable      L equal (v_N/v_rho_mass)^(1.0/3.0)  # Cubic box length

# ---------- Create Simulation Box ----------
region        box block 0 ${L} 0 ${L} 0 ${L} units box
create_box    1 box

# ---------- Place Atoms in Box ----------
create_atoms  1 random ${N} ${seed} box

print         "Computed Box Size (L) in LJ Units: ${L}"

# ---------- Define Lennard-Jones Interactions ----------
pair_style    lj/cut 2.5
pair_coeff    1 1 1.0 1.0 2.5
neighbor      0.3 bin
neigh_modify  every 1 delay 0 check yes
mass          1 1.0

# ---------- Minimization and Equilibration ----------
minimize 1.0e-6 1.0e-8 1000 10000
velocity all create 0.7833 ${seed} mom yes rot yes dist gaussian
fix     equil all nvt temp 0.7833 0.7833 1.0
timestep 0.01
thermo_style custom step temp pe etotal press vol
thermo  10000
run     ${eq_steps}
unfix   equil

# --------- Resetting time step ----------
reset_timestep	0
# ---------- Production Run (NVE Ensemble) ----------
fix           1 all nve
timestep      0.01
thermo_modify lost ignore flush yes
thermo_style  custom step temp pe ke etotal press density
thermo        10000

fix thermo_output all print 1000 "$(step) $(temp) $(pe) $(ke) $(etotal) $(press) $(density)" file thermo_output_${seed}.txt screen no
fix energy_output all print 1000 "$(etotal) $(ke)" file energy_output_${seed}.txt screen no

# ---------- Compute Pair Correlation Function g(r) over entire trajectory ----------
compute       myRDF all rdf 100
fix           RDF_output all ave/time 100 10 1000 c_myRDF[*] file rdf_output_${seed}.txt mode vector

# ---------- Compute Mean Square Displacement (MSD) ----------
compute       MSD all msd
fix           MSD_output all ave/time 100 10 1000 c_MSD[4] file msd_output_${seed}.txt mode scalar

# ---------- Compute Velocity Autocorrelation Function (VACF) over entire trajectory ----------
compute       VACF all vacf
fix           VACF_output all ave/correlate 1 100 100 c_VACF[1] type auto file vacf_output_${seed}.txt ave running overwrite

# ---------- Run Simulation ----------
run           ${run_steps}
```
