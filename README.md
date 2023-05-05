
# Comparison of CNN Architectures

This research compares the capabilities of four distinct CNN models for the task of classifying
medical images: SimpleCNN, LeNet5, AlexNet, and VGG16. The study looks into which CNN
model does this task the best and offers insights into the variables that affect the model's
performance.
- Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Getting Started 
The models are present in the **Neural Network** folder with ipynb file extension.

The link to it: https://github.com/jinunyachhyon/CNN-classification---PyTorch/tree/main/Neural%20Networks

To run this model, install all the requirements with following code: 

```bash
  pip install requirements.txt
```
## Directly use the saved model
Without having to run the models, you can also directly load the saved models present in **Trained models** folder. 

The link to it: https://github.com/jinunyachhyon/CNN-classification---PyTorch/tree/main/Trained%20models

The models are given as: 
- LeNet model:  modelLeNet.pth
- AlexNet model trained without initailized parameters:  modelAlexNet.pth
- AlexNet model trained with initailized parameters:  modelAlexNet_initialparam.pth
- VGG16 model trained without initailized parameters:  modelVGG16.pth
- VGG16 model trained with initailized parameters:  modelVGG16_initialparam.pth

To directly use the saved model, run the code below with **PATH** being the path to the saved model.
```bash
  model = torch.load(PATH)
  model.eval()
```  

## Brief description of the models

- 1- SimpleCNN 
<a target="_blank" href="https://colab.research.google.com/github/jinunyachhyon/CNN-classification---PyTorch/blob/main/Neural%20Networks/SimpleCNN.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The research started with the simplest CNN model which consists of 5 layers where 2
are convolutional layers combined with ReLU activation and 3 are fully connected layers
combined with ReLU activation except the last layer. In this model, the kernel size(filter)
is relatively smaller without stride or padding which resulted in higher number of total
parameters and hence increases the computational cost. This model had a total of
5,406,606 trainable parameters.


- 2 - LeNet5
<a target="_blank" href="https://colab.research.google.com/github/jinunyachhyon/CNN-classification---PyTorch/blob/main/Neural%20Networks/CNN_LeNet.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

LeNet5 is similar to SimpleCNN model that consists of two convolutional layers
combined with ReLU activation, each followed by a subsampling layer, and then three
fully connected linear layers combined with ReLU activation except the last layer. In this
model, the kernel size(filter) is relatively larger than that of SimpleCNN and stride
parameter is also defined which reduced the total parameters and hence, the
computational cost. This model had a total of 150,846 trainable parameters.


- 3 - AlexNet 
<a target="_blank" href="https://colab.research.google.com/github/jinunyachhyon/CNN-classification---PyTorch/blob/main/Neural%20Networks/AlexNet.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

LeNet5 is similar to SimpleCNN model that consists of two convolutional layers
combined with ReLU activation, each followed by a subsampling layer, and then three
fully connected linear layers combined with ReLU activation except the last layer. In this
model, the kernel size(filter) is relatively larger than that of SimpleCNN and stride
parameter is also defined which reduced the total parameters and hence, the
computational cost. This model had a total of 150,846 trainable parameters.


- 4 - VGG16 
<a target="_blank" href="https://colab.research.google.com/github/jinunyachhyon/CNN-classification---PyTorch/blob/main/Neural%20Networks/VGG16.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

VGG16 is another deep CNN model that consists of 16 convolutional layers and three
fully connected layers. The architecture is characterized by a series of small
convolutional filters (3x3) that are stacked together. This allows VGG16 to capture
features at multiple scales and resolutions. However, VGG16 has a large number of
parameters and require more computational resources to train.



