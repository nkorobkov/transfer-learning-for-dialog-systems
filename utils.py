import json
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import contractions
import unicodedata
import re
import time
from collections import defaultdict, Counter


topics = ['BookRestaurant','GetWeather', 'SearchScreeningEvent','RateBook', 'SearchCreativeWork', 'AddToPlaylist', 'PlayMusic']


def import_data(path = 'data/snips_processed/snips.csv'):
    dataset = []
    with open(path, 'r') as f:

        reader = csv.reader(x.replace('\0', '') for x in f)
        for line in reader:
            dataset.append(line)
    dataset = np.array(dataset)

    labels = ['BookRestaurant','GetWeather', 'SearchScreeningEvent','RateBook', 'SearchCreativeWork', 'AddToPlaylist', 'PlayMusic']
    lab2id = {}
    id2lab = {}
    for i in range(len(labels)):
        lab2id[labels[i]] = i
        id2lab[i] = labels[i]
    return dataset, lab2id, id2lab


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    return contractions.fix(text)


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


@np.vectorize
def pre_process_text(document, remove_digits = False):
    # lower case
    document = document.lower()

    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))

    # remove accented characters
    document = remove_accented_chars(document)

    # expand contractions
    document = expand_contractions(document)

    # remove special characters and\or digits
    # insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=remove_digits)

    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()

    return document

def prepare_labs(labs, lab2id):
    out = []
    for lab in labs:
        out.append(lab2id[lab])
    return out

def train(model, criterion, optimizer, labels, vectors):
    model.zero_grad()
    loss = 0

    # vectors = torch.tensor(vectors).float()
    # labels = torch.tensor(labels)

    model_out = model.forward(vectors)
    loss += criterion(model_out[:, 0], labels)

    loss.backward()
    optimizer.step()

    return loss.item() / len(labels)

def evaluate(model, labels, vectors, criterion):
    with torch.no_grad():
        #vectors = torch.tensor(vectors).float()
        #labels = torch.tensor(labels)
    
        model_out = model.forward(vectors)
        right = 0
        
        for i  in range(len(model_out)):
            k, v = model_out[i].topk(1)
            predicted, true = v.item(), labels[i].item()
            if predicted == true:
                right +=1

                
        loss = criterion(model_out[:,0], labels)
        return loss.item(), right/len(model_out)
    
        


class Baseline(nn.Module):
    def __init__(self, in_size = 1024, out_size = 7):
        super(Baseline, self).__init__()

        self.W = nn.Linear(in_size, 7)
        self.out = nn.LogSoftmax(2)
        
    def forward(self, x):
        x = self.W(x)
        return self.out(x)

def get_util(lprop, lload, n):
    a = list(map(lambda x:(x - 1/7)**2, lprop))
    b = lload[lprop.index(max(lprop))]
    
    return sum([aa * bb for aa, bb in zip(a, lload)]) * (7/6)

def compute_per_word_label(labels, sentences):
    en_stats = defaultdict(dict)
    t_stats = Counter(labels)


    for label, line in zip(labels, sentences):
        seen = set()
        for w in line.split():
            if w not in seen:
                en_stats[w][label] = en_stats[w].get(label, 0) + 1
                en_stats[w]['n'] = en_stats[w].get('n', 0) + 1
                seen.add(w)
    utils = []
    for k in en_stats:
        label_prop = []
        label_load = []
        for t in topics:
            en_stats[k][t] = en_stats[k].get(t, 0)
            label_prop.append(en_stats[k][t]/en_stats[k]['n'])
            label_load.append(en_stats[k][t]/t_stats[t])
        en_stats[k]['lprop'] = label_prop
        en_stats[k]['lload'] = label_load
        
        utility = get_util(label_prop, label_load, en_stats[k]['n'])
        utils.append(utility)
        en_stats[k]['u'] = utility
        
    return en_stats, utils


class RNN(nn.Module):
    def __init__(self, max_s_len , emb_dim = 300, out_size = 512):
        super(RNN, self).__init__()
        
        #(batch, sent_len, emb_dim)
        
        self.emb_dim = emb_dim
        self.out_size =out_size
        self.max_s_len = max_s_len
        self.kernel_sizes = [2,3,5]
        self.cnn_chan  = 128
        self.lstm_hid  = 256
        
        self.drop = nn.Dropout(0.35)

        self.max_pool_kernel_size = [(self.max_s_len - x + 1, 1) for x in self.kernel_sizes]

        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.cnn_chan, kernel_size=(self.kernel_sizes[0], self.emb_dim), )
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.cnn_chan, kernel_size=(self.kernel_sizes[1], self.emb_dim), )
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.cnn_chan, kernel_size=(self.kernel_sizes[2], self.emb_dim), )
        
        #dont add dropout  to last layer since there is only one layer
        self.lstm = nn.GRU(input_size = self.emb_dim, hidden_size = self.lstm_hid,
                         batch_first=True, bidirectional=True)
    
    
        self.lin = nn.Linear(self.lstm_hid*2 + self.cnn_chan * 3, out_size)
        
        
        
    def forward(self, x, lens):
        
        # add chanels dimention:
        xc = x.unsqueeze(1)
        
        x1 = self.conv1(xc)
        x1 = F.max_pool2d(F.relu(x1), self.max_pool_kernel_size[0])
        x1 = x1.squeeze(3).squeeze(2)
        
        x2 = self.conv1(xc)
        x2 = F.max_pool2d(F.relu(x2), self.max_pool_kernel_size[1])
        x2 = x2.squeeze(3).squeeze(2)

        x3 = self.conv1(xc)
        x3 = F.max_pool2d(F.relu(x3), self.max_pool_kernel_size[2])
        x3 = x3.squeeze(3).squeeze(2)

        ps = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        
        _, (h) = self.lstm(ps)
        lstm_out = torch.cat((h[1],h[0]), dim =1)

        out = torch.cat((x1,x2,x3, lstm_out), dim=1)
        
        out = self.drop(out)
        out = self.lin(out)
        
        return out
    
    
def closest_index(request, dots, forbiden_index=-1):
    dists = np.linalg.norm(dots-request, axis=1)
    res =  np.argmin(dists)
    
    if res == forbiden_index:
        dists[res] = np.inf
        return np.argmin(dists)
    return res
