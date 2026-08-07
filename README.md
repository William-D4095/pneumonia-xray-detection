# pneumonia-xray-detection
Automated pneumonia detection in chest X-rays using ResNet18







## .onnx generation
the ipynb trains a resnet model and saves it in .onnx format to be used on the orin nanos. (Backup is on email)


## resources
demonstration of this working:
https://drive.google.com/file/d/1T1s5ANJJl2RoLvTtC6Ve-vD5PhXZ59rS/view?usp=sharing
kaggle dataset used to train this model:
https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images/data


run instructions: python3 predict_dev.py --image pneum_lungs.jpg
