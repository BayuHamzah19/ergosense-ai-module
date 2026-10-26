#!/bin/bash
cd ~/Kuliah/ERGO-SENSE
source .venv/bin/activate
for s in 25 26 27 28 29 30 31 33 34 35 36; do
  if [ -f results/stgcn_loso_eyes_$s.csv ]; then echo "lewati $s"; continue; fi
  python src/train_stgcn.py --data data/stgcn_windows_w5_cal_self30_s3.npz \
    --nodes eyes --monitor auc --only-test $s --tag eyes_$s > results/stgcn_eyes_$s.txt 2>&1
done
echo selesai
