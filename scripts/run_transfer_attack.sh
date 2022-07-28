cd ..

modelname=$1
dataname=$2

if [ $dataname = 'CIFAR100' ]
then
    num_classes=100
else
    num_classes=10
fi

python transfer_attack.py \
--savedir ./results/${dataname}/transfer_attack_results \
--exp-name ${modelname} \
--dataname $dataname \
--batch-size 32 \
--num_classes $num_classes \
--known_attack_path ./results/${dataname}/known_attack_results/${modelname}


