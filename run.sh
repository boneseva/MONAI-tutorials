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
export MONAI_DATA_DIRECTORY="/home/bonese/tutorials/3d_segmentation/results"
echo $MONAI_DATA_DIRECTORY

echo "GPUs assigned to this job: $CUDA_VISIBLE_DEVICES"
nvidia-smi  # This will show detailed GPU usage and stats

<<<<<<< HEAD
python 3d_segmentation/swin_unetr_segmentation_3D.py
#python 3d_segmentation/swin_unetr_segmentation_3D_test.py
=======
#python 3d_segmentation/swin_unetr_segmentation_3D.py
python 3d_segmentation/swin_unetr_segmentation_3D_test.py
>>>>>>> 45b62e7 (swin)
