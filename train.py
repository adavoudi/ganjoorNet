from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils
import seq2seq_model
from seq2seq_model import AttnDecoderRNN, EncoderRNN


use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5
MAX_LENGTH = seq2seq_model.MAX_LENGTH
SOS_token = utils.SOS_token
EOS_token = utils.EOS_token

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def save_model(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, ckpt_fname):
    encoder_state_dict = encoder.state_dict()
    for key in encoder_state_dict.keys():
        encoder_state_dict[key] = encoder_state_dict[key].cpu()

    decoder_state_dict = encoder.state_dict()
    for key in decoder_state_dict.keys():
        decoder_state_dict[key] = decoder_state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder_state_dict,
        'decoder_state_dict': decoder_state_dict,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer},
        ckpt_fname)

def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, lines):
    
    # start = time.time()
    # plot_losses = []
    # print_loss_total = 0  # Reset every print_every
    # plot_loss_total = 0  # Reset every plot_every
    criterion = nn.NLLLoss()

    pbar = tqdm(range(1, len(lines)-1))

    for iter in pbar:
        
        training_pair = utils.variablesFromLine(lines[iter])
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        # print_loss_total += loss
        # plot_loss_total += loss

        # if iter % print_every == 0:
        # print_loss_avg = print_loss_total / print_every
        pbar.set_description('%.4f' % loss)
            # print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                              iter, iter / n_iters * 100, print_loss_avg))

        # if iter % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

    # showPlot(plot_losses)


if __name__ == '__main__':

    continue_from_file = None
    hidden_size = 256
    init_epoch = 0
    max_epoch = 4
    save_interval = 1
    save_dir = './models'
    learning_rate=0.01

    encoder = EncoderRNN(utils.ALPHASIZE, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, utils.ALPHASIZE,
                                1, dropout_p=0.1)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    if continue_from_file is not None:
        state_dict = torch.load(continue_from_file)
        encoder_state_dict = OrderedDict()
        for k, value in state_dict['encoder_state_dict'].items():
            key = "module.{}".format(k)
            encoder_state_dict[key] = value
        encoder.load_state_dict(encoder_state_dict)
        decoder_state_dict = OrderedDict()
        for k, value in state_dict['decoder_state_dict'].items():
            key = "module.{}".format(k)
            decoder_state_dict[key] = value
        decoder.load_state_dict(decoder_state_dict)
        encoder_optimizer = state_dict['encoder_optimizer']
        decoder_optimizer = state_dict['decoder_optimizer']
        init_epoch = state_dict['epoch']
        print("pre-trained epoch number: {}".format(init_epoch))

    lines = utils.readPoems('/home/reza/projects/ganjoorNet/ganjoor-scrapy/shahname/*.txt')

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    for epoch in range(init_epoch, max_epoch):
        trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, lines)
        if epoch % save_interval == 0:
            save_model(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, os.path.join(
                save_dir, '%03d.ckpt' % epoch))