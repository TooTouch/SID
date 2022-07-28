# Detecting Adversarial Examples from Sensitivity Inconsistency of Spatial-Transform Domain

Pytorch re-implementation for "Detecting Adversarial Examples from Sensitivity Inconsistency of Spatial-Transform Domain".



# Run

run [`./scripts/run_pipeline.sh`](https://github.com/TooTouch/SID/blob/main/scripts/run_pipeline.sh)


**run_pipeline.sh**

```bash
modelname="resnet34 vgg19"
adv_method_list=("DeepFool" "BIM" "CW" "CW" "FAB" "FGSM" "PGD" "PGD" "PGD")
adv_expname_list=("DeepFool" "BIM" "CW" "Low_CW" "FAB" "FGSM" "PGD" "Low_PGD1" "Low_PGD2")
dataname_list="CIFAR10 SVHN CIFAR100"


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
```

# Results

- **Model**: ResNet34

## 1. Adversarial Attacks

**CIFAR10**

|          |   Adv Acc(%) |   Adv Acc(%) DWT |   # Success Images |
|:---------|-------------:|-----------------:|-------------------:|
| DeepFool |         5.59 |            91.38 |               6528 |
| BIM      |         0.08 |            65.83 |               6968 |
| CW       |        22.66 |            84.85 |               4806 |
| Low_CW   |        58.79 |            92.20 |               1955 |
| FAB      |         0.00 |            92.59 |               6968 |
| FGSM     |        35.80 |            67.45 |               3787 |
| PGD      |         0.06 |            69.14 |               6961 |
| Low_PGD1 |        60.67 |            91.31 |               1681 |
| Low_PGD2 |        14.40 |            88.86 |               5561 |


**CIFAR100**

|          |   Adv Acc(%) |   Adv Acc(%) DWT |   # Success Images |
|:---------|-------------:|-----------------:|-------------------:|
| DeepFool |        15.39 |            72.17 |               2553 |
| BIM      |         0.06 |            47.86 |               3322 |
| CW       |         8.12 |            64.45 |               2577 |
| Low_CW   |        27.78 |            71.95 |               1409 |
| FAB      |         5.10 |            72.41 |               3126 |
| FGSM     |        11.19 |            37.20 |               2300 |
| PGD      |         0.04 |            50.33 |               3294 |
| Low_PGD1 |        32.54 |            70.69 |               1065 |
| Low_PGD2 |         7.35 |            67.55 |               2616 |


**SVHN**

|          |   Adv Acc(%) |   Adv Acc(%) DWT |   # Success Images |
|:---------|-------------:|-----------------:|-------------------:|
| DeepFool |         7.67 |            94.78 |              15362 |
| BIM      |         0.46 |            68.95 |              17027 |
| CW       |        12.61 |            89.32 |              13869 |
| Low_CW   |        41.64 |            95.97 |               6420 |
| FAB      |         1.48 |            96.19 |              16828 |
| FGSM     |        17.43 |            66.65 |              12660 |
| PGD      |         0.54 |            72.63 |              17019 |
| Low_PGD1 |        47.86 |            95.43 |               4884 |
| Low_PGD2 |        20.64 |            93.32 |              11789 |



## 2. Known Attacks 

**CIFAR10**

|          |   AUROC(%) |   Detection Acc(%) | #(train, test)   |
|:---------|-----------:|-------------------:|:-----------------|
| DeepFool |      90.99 |              85.73 | (7979, 5323)     |
| BIM      |      99.14 |              96.33 | (9150, 6103)     |
| CW       |      86.81 |              81.65 | (6017, 4014)     |
| Low_CW   |      85.55 |              79.49 | (2425, 1621)     |
| FAB      |      94.10 |              88.85 | (8544, 5700)     |
| FGSM     |      83.26 |              75.98 | (5060, 3377)     |
| PGD      |      99.22 |              96.28 | (9042, 6030)     |
| Low_PGD1 |      80.72 |              76.90 | (2098, 1403)     |
| Low_PGD2 |      91.26 |              85.59 | (6817, 4547)     |


**CIFAR100**

|          |   AUROC(%) |   Detection Acc(%) | #(train, test)   |
|:---------|-----------:|-------------------:|:-----------------|
| DeepFool |      70.21 |              67.67 | (3208, 2142)     |
| BIM      |      94.62 |              91.79 | (4293, 2866)     |
| CW       |      75.06 |              70.16 | (3221, 2149)     |
| Low_CW   |      73.45 |              70.15 | (1754, 1171)     |
| FAB      |      79.35 |              73.87 | (3875, 2586)     |
| FGSM     |      80.65 |              75.17 | (2939, 1963)     |
| PGD      |      93.85 |              92.00 | (4218, 2816)     |
| Low_PGD1 |      74.59 |              69.40 | (1330, 891)      |
| Low_PGD2 |      73.47 |              72.15 | (3236, 2160)     |



**SVHN**

|          |   AUROC(%) |   Detection Acc(%) | #(train, test)   |
|:---------|-----------:|-------------------:|:-----------------|
| DeepFool |      89.50 |              83.98 | (18876, 12585)   |
| BIM      |      99.89 |              98.81 | (23260, 15512)   |
| CW       |      92.95 |              86.57 | (17276, 11520)   |
| Low_CW   |      89.26 |              84.64 | (7801, 5203)     |
| FAB      |      88.22 |              82.05 | (20413, 13612)   |
| FGSM     |      93.48 |              86.48 | (16737, 11162)   |
| PGD      |      99.86 |              98.48 | (22849, 15238)   |
| Low_PGD1 |      85.86 |              80.71 | (5977, 3988)     |
| Low_PGD2 |      95.90 |              89.29 | (14454, 9641)    |



## 3. Transfer Attacks

- **Row**: Source
- **Column**: Target

**CIFAR10**

|          |   DeepFool |   BIM |    CW |   Low_CW |   FAB |   FGSM |   PGD |   Low_PGD1 |   Low_PGD2 |
|:---------|-----------:|------:|------:|---------:|------:|-------:|------:|-----------:|-----------:|
| DeepFool |      90.99 | 73.41 | 89.18 |    86.67 | 92.34 |  81.68 | 75.21 |      84.81 |      88.85 |
| BIM      |      62.74 | 99.14 | 82.96 |    57.80 | 63.96 |  85.26 | 99.28 |      61.79 |      85.43 |
| CW       |      82.47 | 80.86 | 86.81 |    80.43 | 79.81 |  87.88 | 80.14 |      80.40 |      84.74 |
| Low_CW   |      86.34 | 69.30 | 86.10 |    85.55 | 89.13 |  71.90 | 71.50 |      86.72 |      84.48 |
| FAB      |      90.93 | 77.55 | 88.21 |    81.33 | 94.10 |  77.03 | 77.43 |      76.86 |      88.97 |
| FGSM     |      72.70 | 78.25 | 79.90 |    69.58 | 72.13 |  83.26 | 75.80 |      70.69 |      75.30 |
| PGD      |      64.47 | 99.26 | 82.97 |    59.91 | 66.11 |  86.31 | 99.22 |      63.87 |      85.01 |
| Low_PGD1 |      75.69 | 72.85 | 81.19 |    79.71 | 72.34 |  70.18 | 68.54 |      80.72 |      80.72 |
| Low_PGD2 |      83.62 | 93.92 | 90.43 |    82.03 | 82.55 |  85.97 | 93.34 |      82.85 |      91.26 |


**CIFAR100**

|          |   DeepFool |   BIM |    CW |   Low_CW |   FAB |   FGSM |   PGD |   Low_PGD1 |   Low_PGD2 |
|:---------|-----------:|------:|------:|---------:|------:|-------:|------:|-----------:|-----------:|
| DeepFool |      70.21 | 58.14 | 73.10 |    69.29 | 77.03 |  61.52 | 58.41 |      64.24 |      67.73 |
| BIM      |      23.15 | 94.62 | 47.83 |    34.71 | 32.24 |  40.08 | 94.46 |      40.34 |      58.76 |
| CW       |      68.86 | 74.77 | 75.06 |    73.75 | 73.29 |  72.79 | 76.44 |      67.78 |      79.51 |
| Low_CW   |      86.29 | 80.05 | 88.57 |    73.45 | 84.14 |  66.37 | 78.06 |      83.49 |      88.16 |
| FAB      |      80.78 | 67.59 | 79.75 |    76.49 | 79.35 |  63.70 | 66.82 |      73.45 |      79.91 |
| FGSM     |      64.69 | 64.27 | 76.54 |    64.31 | 60.57 |  80.65 | 59.06 |      57.79 |      58.23 |
| PGD      |      20.76 | 93.79 | 43.67 |    28.38 | 28.88 |  45.01 | 93.85 |      42.11 |      58.10 |
| Low_PGD1 |      72.29 | 77.44 | 77.40 |    78.69 | 78.99 |  54.35 | 83.48 |      74.59 |      90.83 |
| Low_PGD2 |      57.16 | 87.76 | 80.85 |    68.79 | 64.72 |  55.81 | 89.72 |      71.36 |      73.47 |


**SVHN**

|          |   DeepFool |   BIM |    CW |   Low_CW |   FAB |   FGSM |   PGD |   Low_PGD1 |   Low_PGD2 |
|:---------|-----------:|------:|------:|---------:|------:|-------:|------:|-----------:|-----------:|
| DeepFool |      89.50 | 65.65 | 79.25 |    82.53 | 86.18 |  79.43 | 68.55 |      78.52 |      80.19 |
| BIM      |      51.43 | 99.89 | 88.56 |    54.56 | 30.54 |  86.72 | 99.87 |      69.78 |      94.93 |
| CW       |      85.93 | 84.05 | 92.95 |    86.92 | 81.37 |  90.93 | 83.91 |      88.34 |      90.94 |
| Low_CW   |      87.97 | 72.78 | 91.86 |    89.26 | 86.72 |  85.17 | 76.96 |      91.63 |      88.95 |
| FAB      |      89.12 | 67.95 | 80.40 |    83.48 | 88.22 |  77.61 | 68.75 |      81.84 |      79.93 |
| FGSM     |      84.37 | 81.13 | 91.48 |    85.64 | 82.36 |  93.48 | 80.57 |      85.32 |      85.87 |
| PGD      |      54.95 | 99.86 | 90.49 |    58.23 | 32.81 |  88.69 | 99.86 |      72.12 |      95.67 |
| Low_PGD1 |      79.19 | 85.13 | 91.32 |    85.97 | 72.51 |  83.23 | 83.48 |      85.86 |      91.59 |
| Low_PGD2 |      75.87 | 95.21 | 93.79 |    81.86 | 62.61 |  90.29 | 94.70 |      87.53 |      95.90 |


