cd ..

modelname=$1
dataname=$2


# python classifier.py \
# --exp-name ${modelname} \
# --modelname $modelname \
# --dataname $dataname \
# --savedir ./results/${dataname}/saved_model 


# DWT
python classifier.py \
--exp-name ${modelname}_dwt \
--modelname $modelname \
--dataname $dataname \
--num_classes 100 \
--use_wavelet_transform \
--savedir ./results/${dataname}/saved_model 