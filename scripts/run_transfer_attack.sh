cd ..

modelname=$1
dataname=$2

python transfer_attack.py \
--savedir ./results/${dataname}/transfer_attack_results \
--exp-name ${modelname} \
--dataname $dataname \
--batch-size 32 \
--known_attack_path ./results/${dataname}/known_attack_results/${modelname}


