#!/bin/bash

#SBATCH --job-name=yolo-all
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=amd_a100nv_8  # 또는 cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2              # GPU 2장 사용
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00           # ⭐️ 학습+평가 시간을 고려해 넉넉하게 설정 [cite: 864-866]
#SBATCH --comment=pytorch         # 필수 [cite: 468]

# --- 환경 설정 ---
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate neuron-yolo
module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1

# --- 실행 ---
mkdir -p logs
# ⭐️ 통합 스크립트 실행
srun python train_and_eval.py
