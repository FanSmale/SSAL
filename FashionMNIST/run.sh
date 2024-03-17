echo "Running FashionMNIST"
cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train.py \
    --gpu 0 \
    --info F \
    --balance True \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_sup.py \
    --gpu 1 \
    --info F \
    --balance True \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train.py \
    --gpu 1 \
    --info F \
    --balance False \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train.py \
    --gpu 0 \
    --info U \
    --balance True \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_baseline.py \
    --gpu 1 \
    --method Random \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_baseline.py \
    --gpu 1 \
    --method Entropy \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_baseline.py \
    --gpu 1 \
    --method LeastConfidence \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_baseline.py \
    --gpu 1 \
    --method Margin \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_dynamicE2_ssal.py \
    --gpu 1 \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_dynamicE2_baseline.py \
    --gpu 0 \
    --method Random \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_dynamicE2_baseline.py \
    --gpu 0 \
    --method Entropy \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_dynamicE2_baseline.py \
    --gpu 1 \
    --method LeastConfidence \
    > log.out 2>&1 &

nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train_dynamicE2_baseline.py \
    --gpu 1 \
    --method Margin \
    > log.out 2>&1 &
