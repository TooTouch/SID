model_list="resnet34"
adv_method_list="DeepFool"

for modelname in $model_list
do 
    for adv_method in $adv_method_list
    do
        echo "$modelname - $adv_method"
        bash known_attack.sh $modelname $adv_method
    done
done