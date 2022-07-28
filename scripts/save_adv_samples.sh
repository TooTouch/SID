cd ..

modelname=$1
adv_method=$2
adv_name=$3
dataname=$4

if [ $dataname = 'CIFAR100' ]
then
    num_classes=100
else
    num_classes=10
fi

python adv_samples.py \
--savedir ./results/${dataname}/saved_adv_samples \
--exp-name ${modelname}/${adv_name} \
--adv_name $adv_name \
--modelname $modelname \
--dataname $dataname \
--batch-size 64 \
--noise_size 0.05 \
--num_classes $num_classes \
--adv_method $adv_method \
--adv_config ./configs_adv \
--model_checkpoint ./results/${dataname}/saved_model/$modelname/$modelname.pt \
--model_dwt_checkpoint ./results/${dataname}/saved_model/${modelname}_dwt/${modelname}_dwt.pt



