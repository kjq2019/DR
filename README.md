# DR
MA1 project

All results: excels, graphs are in the folder. For V3 there are two versions, I simply seperate the model from scratch and the pretrained one in order to reduce mute and unmute times in code.

The needed data can be downloaded from https://www.kaggle.com/c/aptos2019-blindness-detection/data. We only need train.csv and train_images.

In order to use the python file, V3 is recommended. You still need to manually mute or unmute several lines if you want to load the trained weight.

If you want to use V3_finetune, please first:
pip install git+https://github.com/qubvel/classification_models.git for pretrained ResNet;
pip install -U git+https://github.com/qubvel/efficientnet for pretrained EficientNet.

tensorflow==1.13.1, keras==2.24
