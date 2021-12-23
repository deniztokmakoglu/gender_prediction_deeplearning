"""
Long Short Term Memory Gender Prediction Algorithm
The Economics of Science Project
Project Owner: Ufuk Akçiğit
Author: Deniz Tokmakoglu
"""

import unicodedata
import string
import pandas as pd
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ALL_LET = string.ascii_letters + " .,;'"
N_HIDDEN = 128
N_LET = len(ALL_LET)

class TrainData():
    def __init__(self, data):
        self.data = data.rename(columns = {"Firstname": "name", "Lastname": "name"})
        self.male = self.data[self.data["male"] == 1]
        self.female = self.data[self.data["male"] == 0]
        self.genders = list(data.Gender.unique())
        self.dict_genders = self.generate_dict()
        self.n_categories = len(self.dict_genders)

    def unicodeToAscii(self, s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in ALL_LET
            )
    
    def generate_dict(self):
        dict_gender = {}
        for gender in self.genders:
            data = self.data[self.data.name.notna()]
            name_temp = list(data[data.Gender == gender].name.unique())
            dict_gender[gender] = [self.unicodeToAscii(name) for name in name_temp]
        return dict_gender

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def train_helper(category_tensor, line_tensor, learning_rate, rnn_object, criterion):
    hidden = rnn_object.initHidden()
    rnn_object.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_object(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn_object.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def train(n_iters, print_every, plot_every, gender_dict, genders, learning_rate, rnn_object):
    criterion = nn.NLLLoss()
    current_loss = 0
    all_losses = []
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(gender_dict, genders)
        output, loss = train_helper(category_tensor, line_tensor, learning_rate, rnn_object, criterion)
        current_loss += loss
    # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, genders)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    plt.figure()
    plt.plot(all_losses)
    


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(gender_dict, genders):
    category = randomChoice(genders)
    line = randomChoice(gender_dict[category])
    category_tensor = torch.tensor([genders.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def letterToIndex(letter):
    return ALL_LET.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, N_LET)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, N_LET)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output, genders):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return genders[category_i], category_i


   

def evaluate(line_tensor, rnn_object):
    hidden = rnn_object.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_object(line_tensor[i], hidden)

    return output



def predict(input_line, genders, rnn_object, n_predictions=1):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line), rnn_object)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, genders[category_index]))
            predictions.append([value, genders[category_index]])
            
def predict_2(input_line, genders, rnn_object, n_predictions=3):
     with torch.no_grad():
        output = evaluate(lineToTensor(input_line), rnn_object)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append(genders[category_index])
        return predictions
    
def plot_confisuon(genders, rnn_object, gender_dict, n_confusion):
    confusion = torch.zeros(len(genders), len(genders))
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(gender_dict, genders)
        output = evaluate(line_tensor, rnn_object)
        guess, guess_i = categoryFromOutput(output, genders)
        category_i = genders.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(genders)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + genders, rotation=90)
    ax.set_yticklabels([''] + genders)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

