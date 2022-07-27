cd ..

modelname=$1
dataname=$2

python classifier.py \
--exp-name ${modelname}_dwt \
--modelname $modelname \
--dataname $dataname \
--use_wavelet_transform \
--savedir ./results/${dataname}/saved_model 