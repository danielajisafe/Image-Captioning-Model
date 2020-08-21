# Image-Captioning-Model

In this project, I demonstrated my knowledge of deep learning architectures in an encoder-decoder fashion. The encoder is a convolutional neural network that extracts spatial features as input to the decoder while the decoder is a recurrent neural network that extracts temporal information.

Technically, The Encoder uses a pretrained resnet model to output feature vectors for each input image. The output feature vector passes through an embedding layer. A batch normalization is applied to standardize the embedding before being passed as input to the decoder. The embedded feature vector now becomes the first input to the LSTM cell that produces the hidden state. The hidden state is then passed to a linear layer that outputs the predicted word which feeds back to the LSTM cell in a recurrent manner. The final output is a sequence of words that describes the content in the image.

The complete architecture was trained on (MS COCO) dataset and was tested on novel images to automatically generate captions!

![Model Prediction](/images/prediction.png)  

Paper References: 
1) https://arxiv.org/pdf/1411.4555.pdf
2) https://arxiv.org/pdf/1502.03044.pdf

@Udacity Computer Vision Nanodegree
