import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        #batch normalization
        self.bn1 = nn.BatchNorm1d(embed_size)
        
        #weights initialization for the embedding layer
        self.embed.weight.data.normal_(0,0.02) #small guassian distributed values
        self.embed.bias.data.fill_(0) #zeros

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
  
        #embed feature vectors + batch norm
        features = self.embed(features)
        features = self.bn1(features)
 
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        #define properties
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        #Defining the architecture, inspired from the original paper
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size,self.num_layers, batch_first = True) # hidden outputs
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)# converts to vocab size, then specified embed size
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size) #vector of vocab size
        #we need the logits, so no softmax here
        
        #weight initialization for the decoder
        self.embeddings.weight.data.uniform_(-0.1,0.1)
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)
        
        
    
    def forward(self, features, captions): #for the training phase
        #e.g input feature -> (10,512), caption -> (10, 14)
        
        #Now embed the captions
        caption_embed = self.embeddings(captions)
        #stacking image features and caption embeddings in 1d array
        features = features.unsqueeze(1)
        all_embeddings = torch.cat((features, caption_embed[:,:-1,:]), dim =1) #captions shifted
        
        #LSTM
        hiddens , c = self.lstm(all_embeddings)
        
        #linear that feeds to the next LSTM cell and also contains the previous state
        outputs = self.linear(hiddens)
        
        return outputs
        
        
    def sample(self, inputs, states=None, max_len=20): #for the testing phase, so no targets
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sampled_ids = []
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)
            logits = self.linear(hidden.squeeze(1))
            
            _, predicted = logits.max(dim = 1)
            sampled_ids.append(predicted)
            
            if predicted == 1: #encounters <end>
                break
            
            #update hidden state with new output for the next LSTM cell
            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1)
            
        sampled_ids = torch.stack(sampled_ids, 1)
        sampled_ids = list(sampled_ids.cpu().numpy()[0])
        sampled_ids = [int(i) for i in sampled_ids]
        return sampled_ids
            