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

