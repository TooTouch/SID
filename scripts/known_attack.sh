cd ..

modelname=$1
adv_method=$2
dataname=$3

python known_attack.py \
--savedir ./results/${dataname}/known_attack_results \
--exp-name ${modelname}/${adv_method} \
--modelname $modelname \
--dataname $dataname \
--epochs 200 \
--batch-size 32 \
--save_bucket_path ./results/${dataname}/saved_adv_samples/${modelname}/${adv_method}/successed_images.pkl \
--model_checkpoint ./results/${dataname}/saved_model/${modelname}/${modelname}.pt \
--model_dwt_checkpoint ./results/${dataname}/saved_model/${modelname}_dwt/${modelname}_dwt.pt



