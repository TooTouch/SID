# Detecting Adversarial Examples from Sensitivity Inconsistency of Spatial-Transform Domain

Pytorch re-implementation for "Detecting Adversarial Examples from Sensitivity Inconsistency of Spatial-Transform Domain".



# Run

run [`./scripts/run_pipeline.sh`](https://github.com/TooTouch/SID/blob/main/scripts/run_pipeline.sh)


**run_pipeline.sh**

```bash
modelname="resnet34"
adv_method_list=("DeepFool" "BIM" "CW" "CW" "FAB" "FGSM" "PGD" "PGD" "PGD")
adv_expname_list=("DeepFool" "BIM" "CW" "Low_CW" "FAB" "FGSM" "PGD" "Low_PGD1" "Low_PGD2")
dataname_list="CIFAR10"


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
    bash transfer_attack.sh $modelname $dataname 

done
```

# Results

## 1. Adversarial Attacks

**CIFAR10**

|          | Adv Acc(%) | Adv Acc (%) DWT |
|:---------|-----------:|----------------:|
| DeepFool |       5.59 |           91.38 |
| BIM      |       0.08 |           65.83 |
| CW       |      22.66 |           84.85 |
| Low_CW   |      58.79 |           92.20 |
| FAB      |       0.00 |           92.59 |
| FGSM     |      35.80 |           67.45 |
| PGD      |       0.06 |           69.14 |
| Low_PGD1 |      60.67 |           91.31 |
| Low_PGD2 |      14.40 |           88.86 |



## 2. Known Attacks 

**CIFAR10**

|          |  AUROC(%) |   Detection Acc(%) |
|:---------|----------:|-------------------:|
| DeepFool |     90.99 |              85.73 |
| BIM      |     99.14 |              96.33 |
| CW       |     86.81 |              81.65 |
| Low_CW   |     85.55 |              79.49 |
| FAB      |     94.10 |              88.85 |
| FGSM     |     83.26 |              75.98 |
| PGD      |     99.22 |              96.28 |
| Low_PGD1 |     80.72 |              76.90 |
| Low_PGD2 |     91.26 |              85.59 |


## 3. Transfer Attacks

**CIFAR10**

|          |   DeepFool |   BIM |    CW |   Low_CW |   FAB |   FGSM |   PGD |   Low_PGD1 |   Low_PGD2 |
|:---------|-----------:|------:|------:|---------:|------:|-------:|------:|-----------:|-----------:|
| DeepFool |      90.99 | 73.41 | 89.18 |    86.67 | 92.34 |  81.68 | 75.21 |      84.81 |      88.85 |
| BIM      |      62.74 | 99.14 | 82.96 |    57.80 | 63.96 |  85.26 | 99.28 |      61.79 |      85.43 |
| CW       |      82.47 | 80.86 | 86.81 |    80.43 | 79.81 |  87.88 | 80.14 |      80.40 |      84.74 |
| Low_CW   |      86.34 | 69.3  | 86.10 |    85.55 | 89.13 |  71.90 | 71.50 |      86.72 |      84.48 |
| FAB      |      90.93 | 77.55 | 88.21 |    81.33 | 94.10 |  77.03 | 77.43 |      76.86 |      88.97 |
| FGSM     |      72.70 | 78.25 | 79.90 |    69.58 | 72.13 |  83.26 | 75.80 |      70.69 |      75.30 |
| PGD      |      64.47 | 99.26 | 82.97 |    59.91 | 66.11 |  86.31 | 99.22 |      63.87 |      85.01 |
| Low_PGD1 |      75.69 | 72.85 | 81.19 |    79.71 | 72.34 |  70.18 | 68.54 |      80.72 |      80.72 |
| Low_PGD2 |      83.62 | 93.92 | 90.43 |    82.03 | 82.55 |  85.97 | 93.34 |      82.85 |      91.26 |