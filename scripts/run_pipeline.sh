modelname_list="vgg19 resnet34"
adv_method_list=("DeepFool" "BIM" "CW" "CW" "FAB" "FGSM" "PGD" "PGD" "PGD")
adv_expname_list=("DeepFool" "BIM" "CW" "Low_CW" "FAB" "FGSM" "PGD" "Low_PGD1" "Low_PGD2")
dataname_list="CIFAR10 SVHN CIFAR100"


for modelname in $modelname_list
do
    for dataname in $dataname_list
    do
        # 1. train classifier
        bash run_classifier.sh $modelname $dataname 

        # 2. make adversarial examples
        for i in ${!adv_method_list[*]}
        do
            bash save_adv_samples.sh $modelname ${adv_method_list[$i]} ${adv_expname_list[$i]} $dataname
        done

        # 3. known attack
        for i in ${!adv_method_list[*]}
        do
            bash known_attack.sh $modelname ${adv_expname_list[$i]} $dataname
        done

        # 4. transfer attack
        bash run_transfer_attack.sh $modelname $dataname 

    done
done