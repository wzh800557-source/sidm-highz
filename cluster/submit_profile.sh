#!/bin/bash
#SBATCH --job-name=sidm_profile
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=profile_scan_%j.log
#SBATCH --error=profile_scan_%j.err

echo "=== SIDM Profile Scan ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo ""

# Environment
export PATH=~/miniconda3/bin:$PATH

# Run
cd ~/cluster_profile
python3 -u run_profile_scan.py 2>&1

echo ""
echo "End: $(date)"
