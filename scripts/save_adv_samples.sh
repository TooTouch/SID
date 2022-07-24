cd ..

modelname=$1
adv_method=$2

python adv_samples.py \
--savedir ./saved_adv_samples \
--exp-name ${modelname}/${adv_method} \
--modelname $modelname \
--dataname CIFAR10 \
--batch-size 64 \
--noise_size 0.01 \
--adv_method $adv_method \
--adv_config ./configs_adv \
--model_checkpoint ./saved_model/$modelname/$modelname.pt \
--model_dwt_checkpoint ./saved_model/${modelname}_dwt/${modelname}_dwt.pt



