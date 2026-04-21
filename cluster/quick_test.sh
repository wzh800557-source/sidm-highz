#!/bin/bash
# Quick test: run reduced grid locally (~20 min) to verify everything works
# before submitting to cluster.
#
# Usage: bash quick_test.sh

echo "=== Quick Profile Scan Test ==="
echo "This runs a reduced 7×5 = 35 point grid (~20 min)."
echo "Full scan (18×14 = 252 points) should be run on cluster."
echo ""

python3 -u run_profile_scan.py --quick 2>&1 | tee quick_test.log

echo ""
echo "If this succeeded, upload to cluster and run:"
echo "  scp -r cluster_profile/ wzh557@orcd-login002:~/"
echo "  ssh wzh557@orcd-login002"
echo "  cd ~/cluster_profile && sbatch submit_profile.sh"
