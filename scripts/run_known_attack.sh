cd ..

modelname=$1
adv_method=$2

python known_attack.py \
--savedir ./known_attack_results \
--exp-name ${modelname}/${adv_method} \
--modelname $modelname \
--dataname CIFAR10 \
--batch-size 64 \
--save_bucket_path ./saved_adv_samples/${modelname}/${adv_method}/successed_images.pkl \
--model_checkpoint ./saved_model/${modelname}/${modelname}.pt \
--model_dwt_checkpoint ./saved_model/${modelname}_dwt/${modelname}_dwt.pt



