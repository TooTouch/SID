model_list="resnet34"
adv_method_list=("DeepFool" "BIM" "CW" "CW" "FAB" "FGSM" "PGD" "PGD" "PGD")
adv_expname_list=("DeepFool" "BIM" "CW" "Low_CW" "FAB" "FGSM" "PGD" "Low_PGD1" "Low_PGD2")
dataname_list="CIFAR10 CIFAR100 SVHN"


for dataname in $dataname_list
do
    for modelname in $model_list
    do 
        for adv_method in $adv_method_list
        do
            bash save_adv_samples.sh $modelname $adv_method $dataname
        done
    done
done