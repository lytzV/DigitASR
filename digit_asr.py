# import comet_ml at the top of your file
from comet_ml import Experiment
import torch
from torch import nn
import torchaudio
import requests
import matplotlib.pyplot as plt
import glob
import numpy as np
import itertools
import torch.nn.functional as F


# Create an experiment with your api key:
experiment = Experiment(
    api_key="lq1LdmJ5JGrPZ5bZWCP7CsndO",
    project_name="digit-recognition",
    workspace="lytzv",
)



class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self.class_size = 12
        self.batch_size = 1
        self.max_target_seq_len = 3
        self.blank = self.class_size - 1
        self.data_augmentation_scale = 50
        self.n_mels = 128
        self.print_loss = 100
        
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2).double()  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.n_cnn_layers = 3
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=0.1, n_feats=self.n_mels) 
            for _ in range(self.n_cnn_layers)
        ])
        self.encoder = Encoder(self.n_mels)
        self.decoder = Decoder(self.encoder.hidden_size, self.class_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_mels, self.n_mels), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_mels, self.class_size)
        ).double()
        self.optimizer = torch.optim.SGD(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.001)
        self.loss = nn.CTCLoss(blank=self.blank)

    def run(self):
        training_data, training_truth, training_lengths, training_target_lengths, test_data, test_truth, test_lengths, test_target_lengths  = self.load_data()

        # start training
        training_data, training_truth, training_lengths, training_target_lengths = torch.split(torch.from_numpy(training_data), self.batch_size), torch.split(torch.from_numpy(training_truth), self.batch_size), torch.split(torch.from_numpy(training_lengths), self.batch_size), torch.split(torch.from_numpy(training_target_lengths), self.batch_size)

        print("Start training with {} data points...".format(len(training_data)))
        
        
        for i,t in enumerate(training_data):
            spectro, ground_truth = t.reshape(self.batch_size, self.n_mels, -1), training_truth[i].reshape(self.batch_size, self.max_target_seq_len)
            #cnn_output = self.cnn(spectro.unsqueeze(1))
            #recurrent_cnn_output = self.rescnn_layers(cnn_output)
            #print(recurrent_cnn_output.size())
            encoder_output = self.encoder(spectro)[0]  #output: SpectroTime*N*hidden_size
            posterior = F.log_softmax(self.classifier(encoder_output.double()), dim=2)

            #posterior = self.decoder(encoder_output)
            
            input_length = training_lengths[i]
            target_length = training_target_lengths[i]
            
            self.optimizer.zero_grad()
            loss = self.loss(posterior, ground_truth, input_length, target_length)
            loss.backward()
            experiment.log_metric('loss', loss.item())

            if i%self.print_loss == 0:
                print("loss:", loss.item())
                predictions = self.greedy_reconstruction(posterior)
                for i,p in enumerate(predictions[0:2]):
                    print("should be {} & predicted {}".format([t for t in ground_truth[i].detach().numpy() if t != self.blank], p))

            
            
            self.optimizer.step()
    
    def greedy_reconstruction(self, posterior):
        """
        posterior: SpectroTime*N*C

        results: N*ragged shape
        """
        most_lilely = np.argmax(posterior.detach().numpy(), axis=2).T #N*SpectroTime
       
        results = []
        for prediction in most_lilely:
            prediction = prediction.squeeze()
            seen = None
            result = []
            for p in prediction:
                if p == self.blank:
                    seen = None
                elif p != seen:
                    result.append(p)
                    seen = p 
            results.append(result)
        return results

    def load_data(self):
        data = []
        truths = []
        input_lengths = []
        truth_lengths = []
        max_T = -1

        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=self.n_mels),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

        for filepath in glob.iglob('data/*.wav'):
            asterisk_loc = filepath.find('*')
            asterisk_loc = -4 if asterisk_loc == -1 else asterisk_loc
            ground_truth = [int(s) for s in filepath[5:asterisk_loc].split('-')]
            if ground_truth[1] != self.blank or ground_truth[2] != self.blank: # only training on single digits
                continue
            
            waveform, sample_rate = torchaudio.load(filepath)

            #TODO: This is needs to be changed as it is applied to test data as well but we are not using those at the moment
            for i in range(self.data_augmentation_scale):
                spectrogram = train_audio_transforms(waveform)

                data.append(spectrogram)
                truths.append(ground_truth)
                try:
                    first_blank = ground_truth.index(self.blank)
                except:
                    first_blank = -1
                target_length = first_blank if first_blank != -1 else self.max_target_seq_len
                truth_lengths.append(target_length)
                input_lengths.append(spectrogram.shape[2])

                if spectrogram.shape[2] > max_T:
                    max_T = spectrogram.shape[2]
        
        padded_data = np.zeros((len(data), self.n_mels, max_T))
        for i,d in enumerate(data):
            padded_data[i:i+1, :, 0:d.shape[2]] = d.numpy()
        data = padded_data

        shuffler = np.random.permutation(len(data))
        data = np.array(data)[shuffler]
        truths = np.array(truths)[shuffler]
        input_lengths = np.array(input_lengths)[shuffler]
        truth_lengths = np.array(truth_lengths)[shuffler]

        split_index = int(len(data)*0.9)
        training_data, training_truth, training_lengths, training_target_lengths, test_data, test_truth, test_lengths, test_target_lengths = data[:split_index, :], truths[:split_index, :], input_lengths[:split_index], truth_lengths[:split_index], data[split_index:, :], truths[split_index:, :], input_lengths[split_index:], truth_lengths[split_index:]
        return training_data, training_truth, training_lengths, training_target_lengths, test_data, test_truth, test_lengths, test_target_lengths

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.sample_rate = 16000 # default
        self.feature_extrator = torchaudio.transforms.MFCC(sample_rate=self.sample_rate).double()
        self.hidden_size = hidden_size#self.feature_extrator.n_mfcc
        self.gru = nn.GRU(self.hidden_size, self.hidden_size).double()
        self.layer_norm = nn.LayerNorm(self.hidden_size).double()
        
    def forward(self, input):
        # this will take in waveform
        """
            input: N*hidden*SpectroTime
            hidden: N*1*hidden

            output: SpectroTime*N*hidden_size
            last_hidden: gru_layer*N*hidden_size
        """
        N, T = input.shape[0], input.shape[2]
        feature_vectors = input.transpose(1,2)#self.feature_extrator(input.double())
        feature_vectors = self.layer_norm(feature_vectors)
        feature_vectors = F.gelu(feature_vectors)
        
        feature_vectors = feature_vectors.reshape((-1, N, self.hidden_size)) #SpectroTime*N*hidden_size
        
        encoder_hidden = self.initHidden(N)

        output, last_hidden = self.gru(feature_vectors.double(), encoder_hidden.double()) 

        return output, last_hidden #SpectroTime*N*hidden_size, gru_layer*N*hidden_size
    
    def initHidden(self, n):
        return torch.zeros(self.gru.num_layers, n, self.hidden_size) #gru_layer*N*hidden_size

class Decoder(nn.Module):
    def __init__(self, hidden_size, class_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.class_size = class_size
        self.gru = nn.GRU(self.hidden_size, self.class_size).double()

    def forward(self, encoder_output):
        """
            encoder_output: SpectroTime*N*hidden_size

            output: SpectroTime*N*C
        """
        
        decoder_hidden = self.initHidden(encoder_output.shape[1]) #SpectroTime*N*hidden_size
        output, last_hidden = self.gru(encoder_output.double(), decoder_hidden.double()) #SpectroTime*N*C
        posterior = nn.functional.log_softmax(output, dim=2)
        
        return posterior
    
    def initHidden(self, n):
        return torch.zeros(self.gru.num_layers, n, self.class_size) #gru_layer*N*hidden_size

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


if __name__ == "__main__":
    model = Model()
    model.run()


