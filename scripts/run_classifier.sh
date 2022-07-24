cd ..

python classifier.py --exp-name 'vgg19' --modelname vgg19 
python classifier.py --exp-name 'vgg19_dwt' --modelname vgg19 --use_wavelet_transform

python classifier.py --exp-name 'resnet34' --modelname resnet34 
python classifier.py --exp-name 'resnet34_dwt' --modelname resnet34 --use_wavelet_transform