# Image-Captioning

Project 2: IMAGE CAPTIONING MODEL

In this project, I demonstrated my knowledge of deep learning architectures in an encoder-decoder fashion. The encoder is a convolutional neural network that extracts spatial features as input to the decoder while the decoder is a recurrent network that extracts temporal information.

Technically, The Encoder uses a pretrained resnet model to output feature vectors for each image. The feature vector passes through an embedding layer. A batch normalization is applied to standardize the embedding before being passed as input to the decoder. The embedded feature vector becomes the first input to the LSTM cell that produces the hidden state. The hidden state is passed to the linear layer that outputs the predicted word which feeds back to the LSTM cell in a recurrent manner. The final output is a sequence of words that describes the content in the image.

The complete architecture automatically generate captions from images.

The full network was trained on (MS COCO) dataset and was tested on novel images to automatically generate captions!

Paper References: 
1) https://arxiv.org/pdf/1411.4555.pdf
2) https://arxiv.org/pdf/1502.03044.pdf
