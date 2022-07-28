cd ..

modelname=$1
adv_method=$2
dataname=$3

if [ $dataname = 'CIFAR100' ]
then
    num_classes=100
else
    num_classes=10
fi

python known_attack.py \
--savedir ./results/${dataname}/known_attack_results \
--exp-name ${modelname}/${adv_method} \
--modelname $modelname \
--dataname $dataname \
--epochs 200 \
--batch-size 32 \
--num_classes $num_classes \
--save_bucket_path ./results/${dataname}/saved_adv_samples/${modelname}/${adv_method}/successed_images.pkl \
--model_checkpoint ./results/${dataname}/saved_model/${modelname}/${modelname}.pt \
--model_dwt_checkpoint ./results/${dataname}/saved_model/${modelname}_dwt/${modelname}_dwt.pt



