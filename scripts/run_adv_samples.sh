model_list="resnet34"
adv_method_list="FGSM BIM DeepFool CW"

for modelname in $model_list
do 
    for adv_method in $adv_method_list
    do
        bash save_adv_samples.sh $modelname $adv_method
    done
done