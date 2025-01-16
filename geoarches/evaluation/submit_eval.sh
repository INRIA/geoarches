#!/bin/bash
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --partition=gpu                   # GPU

##########################################
###### run rank_hist ######
##########################################
#SBATCH --job-name=eval_hist_aflow        # Job name
#SBATCH --mem=60G                         # Total memory allocated
#SBATCH --output=eval_hist_aflow_%j.out   # Output file (with job ID)
#SBATCH --error=eval_hist_aflow_%j.err    # Error file (with job ID)
#SBATCH --time=9:00:00                    # Time limit (9 hours)

python -m geoarches.evaluation.eval_multistep  \
--pred_path /scratch/gcouairo/evalstore/jz-geodiff-awflow-m4o-ftval/ \
--output_dir /scratch/resingh/weather/evaluation/awflow/ \
--groundtruth_path data/era5_240/full/ \
--multistep 10 --year 2020 \
--num_workers 4 \
--metrics era5_rank_histogram_25_members