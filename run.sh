#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J swin_unetr_segmentation_3D
#SBATCH -o /home/bonese/log/swin_unetr_segmentation_3D-%J.out
#SBATCH -e /home/bonese/log/swin_unetr_segmentation_3D-%J.err
#SBATCH --mail-user=eva.bones@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:a100:1

source /home/bonese/miniconda3/etc/profile.d/conda.sh
conda activate MONAI
echo "...activated the conda environment..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use SLURM_JOB_ID to construct the directory name
export MONAI_DATA_DIRECTORY="/home/bonese/tutorials/3d_segmentation/results${SLURM_JOB_ID}"
echo $MONAI_DATA_DIRECTORY

mkdir -p $MONAI_DATA_DIRECTORY  # Added -p to avoid errors if the directory already exists

echo "GPUs assigned to this job: $CUDA_VISIBLE_DEVICES"
nvidia-smi  # This will show detailed GPU usage and stats

python 3d_segmentation/swin_unetr_segmentation_3D.py
# python 3d_segmentation/swin_unetr_segmentation_3D_test.py
