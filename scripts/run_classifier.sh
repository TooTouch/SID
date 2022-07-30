cd ..

modelname=$1
dataname=$2

# num classes
if [ $dataname = 'CIFAR100' ]
then
    num_classes=100
else
    num_classes=10
fi

# epochs
if [ $modelname = 'resnet34' ]
then
    epochs=300
else
    epochs=200
fi


python classifier.py \
--exp-name ${modelname} \
--modelname $modelname \
--dataname $dataname \
--num_classes $num_classes \
--epochs $epochs \
--savedir ./results/${dataname}/saved_model \
--checkpoint ./results/${dataname}/saved_model/${modelname}/${modelname}.pt


# DWT
python classifier.py \
--exp-name ${modelname}_dwt \
--modelname $modelname \
--dataname $dataname \
--num_classes $num_classes \
--epochs $epochs \
--use_wavelet_transform \
--savedir ./results/${dataname}/saved_model \
--checkpoint ./results/${dataname}/saved_model/${modelname}_dwt/${modelname}_dwt.pt