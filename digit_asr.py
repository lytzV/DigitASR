import torch
from torch import nn
import torchaudio
import requests
import matplotlib.pyplot as plt
import glob
import numpy as np



class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.class_size = 12
        self.batch_size = 1
        self.max_target_seq_len = 3
        self.blank = self.class_size - 1
        self.decoder = Decoder(self.encoder.hidden_size, self.class_size)
        self.optimizer = torch.optim.Adam([self.encoder.parameters(), self.decoder.parameters()])
        self.loss = nn.CTCLoss(blank=self.blank)
        
    def run(self):
        training_data, training_truth, test_data, test_truth = self.load_data()

        # start training
        training_data, training_truth = torch.split(training_data, self.batch_size), torch.split(training_truth, self.batch_size)
        
        # let's keep it sgd for now so we don't have to worry about T being not the same for a batch
        for i,t in enumerate(training_data):
            t = np.array(t)
            waveform, ground_truth = torch.tensor(t).reshape(self.batch_size, 1, -1), torch.tensor(training_truth[i]).reshape(self.batch_size, self.max_target_seq_len)
            encoder_output = self.encoder(waveform)[0]
            posterior = self.decoder(encoder_output)
            input_length = torch.tensor([waveform.shape[2]])

            first_blank = training_truth[i].index(self.blank)
            target_length = torch.tensor([first_blank if first_blank != -1 else self.max_target_seq_len])
            self.optimizer.zero_grad()
            loss = self.loss(posterior, ground_truth)
            print("loss:", loss)
            self.optimizer.step()
            
    def load_data(self):
        data = []
        truths = []

        for filepath in glob.iglob('data/*.wav'):
            ground_truth = filepath[5:-4].split('-')
            waveform, sample_rate = torchaudio.load(filepath)
            data.append(waveform)
            truths.append(ground_truth)
    

        shuffler = np.random.permutation(len(data))
        data = data[shuffler]
        truths = truths[shuffler]

        split_index = int(len(data)*0.9)
        training_data, training_truth, test_data, test_truth = data[:split_index, :], truths[:split_index, :], data[split_index:, :], truths[split_index:, :]
        return training_data, training_truth, test_data, test_truth

    #def normalize(tensor):
    #    # Subtract the mean, and scale to the interval [-1,1]
    #    tensor_minusmean = tensor - tensor.mean()
    #    return tensor_minusmean/tensor_minusmean.abs().max()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature_extrator = torchaudio.transforms.MFCC()
        self.hidden_size = self.feature_extrator.n_mfcc
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        
    def forward(self, input):
        # this will take in waveform
        """
            input: N*1*T
            hidden: N*1*hidden

            output: T*N*hidden_size
            last_hidden: gru_layer*N*hidden_size
        """
        N, T = input.shape[0], input.shape[2]
        feature_vectors = self.feature_extrator(input).reshape((T, N, self.hidden_size)) #T*N*hidden_size

        encoder_hidden = self.initHidden(N)
        output, last_hidden = self.gru(feature_vectors, encoder_hidden) 

        return output, last_hidden #T*N*hidden_size, gru_layer*N*hidden_size
    
    def initHidden(self, n):
        return torch.zeros(self.gru.num_layers, n, self.hidden_size) #gru_layer*N*hidden_size

class Decoder(nn.Module):
    def __init__(self, hidden_size, class_size):
        super(Decoder, self).__init__()
        self.decoder = self.build_mlp(hidden_size, class_size, 3, 32)
        self.class_size = class_size

        
    def forward(self, encoder_output):
        """
            encoder_output: T*N*hidden_size

            output: T*N*C
        """
        posterior = torch.zeros(encoder_output.shape[0], encoder_output.shape[1], self.class_size, requires_grad=True)
        for t in encoder_output:
            posterior[t,:,:] = self.decoder(t)
        
        return posterior

    def build_mlp(self,
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = nn.Tanh(),
        output_activation = nn.LogSoftmax()):

        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(output_activation)
        return nn.Sequential(*layers)