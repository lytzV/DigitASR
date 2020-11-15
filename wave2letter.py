# import comet_ml at the top of your file
from comet_ml import Experiment
import torch
from torch import nn
import torchaudio
from torchaudio import models
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
        self.print_loss = 5
        self.n_feature = 0
        

        # n residual cnn layers with filter size of 32
        self.wav2letter = models.Wav2Letter(num_classes=self.class_size, input_type='mfcc', num_features=self.n_feature)#.double()
        #self.cov = models.ConvTasNet(num_sources=self.class_size).double()
        self.optimizer = torch.optim.SGD(itertools.chain(self.wav2letter.parameters()), lr=0.001)
        self.loss = nn.CTCLoss(blank=self.blank)

    def run(self):
        training_data, training_truth, training_lengths, training_target_lengths, test_data, test_truth, test_lengths, test_target_lengths = self.load_data()

        # start training
        training_data, training_truth, training_lengths, training_target_lengths = torch.split(torch.from_numpy(training_data), self.batch_size), torch.split(torch.from_numpy(training_truth), self.batch_size), torch.split(torch.from_numpy(training_lengths), self.batch_size), torch.split(torch.from_numpy(training_target_lengths), self.batch_size)

        print("Start training with {} data points...".format(len(training_data)))
        
        
        for i,t in enumerate(training_data):
            waveform, ground_truth = t.reshape(self.batch_size, self.n_feature, -1), training_truth[i].reshape(self.batch_size, self.max_target_seq_len)
            print(waveform.shape)
            posterior = self.wav2letter(waveform).permute(2,0,1)
            #posterior = self.cov(waveform).permute(2,0,1)
            print(posterior.shape)
            
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


        for filepath in glob.iglob('data/*.wav'):
            asterisk_loc = filepath.find('*')
            asterisk_loc = -4 if asterisk_loc == -1 else asterisk_loc
            ground_truth = [int(s) for s in filepath[5:asterisk_loc].split('-')]
            if ground_truth[1] != self.blank or ground_truth[2] != self.blank: # only training on single digits
                continue
            
            waveform, sample_rate = torchaudio.load(filepath)
            mfcc = torchaudio.transforms.MFCC()
            self.n_feature = mfcc.n_mfcc
            waveform = mfcc(waveform)

            data.append(waveform)
            truths.append(ground_truth)
            try:
                first_blank = ground_truth.index(self.blank)
            except:
                first_blank = -1
            target_length = first_blank if first_blank != -1 else self.max_target_seq_len
            truth_lengths.append(target_length)
            input_lengths.append(waveform.shape[2])

            if waveform.shape[2] > max_T:
                max_T = waveform.shape[2]
        
        padded_data = np.zeros((len(data), self.n_feature, max_T))
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

if __name__ == "__main__":
    model = Model()
    model.run()


