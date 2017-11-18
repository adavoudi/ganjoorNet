from __future__ import print_function
import numpy as np
import glob
import sys
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
# size of the alphabet that we work with
ALPHASIZE = 38
SOS_token = 36
EOS_token = 37
MIN_WORDS = 3
MAX_WORDS = 6

farsiToPingilish = {
    0 : 'a',
    1 : 'a',
    2 : 'b',
    3 : 'p',
    4 : 't',
    5 : 's',
    6 : 'j',
    7 : 'ch',
    8 : 'h',
    9 : 'kh',
    10 : 'd',
    11 : 'z',
    12 : 'r',
    13 : 'z',
    14 : 'zh',
    15 : 's',
    16 : 'sh',
    17 : 's',
    18 : 'z',
    19 : 't',
    20 : 'z',
    21 : 'e',
    22 : 'gh',
    23 : 'f',
    24 : 'gh',
    25 : 'k',
    26 : 'g',
    27 : 'l',
    28 : 'm',
    29 : 'n',
    30 : 'v',
    31 : 'h',
    32 : 'y',
    33 : 'y',
    34 : '\n',
    35 : '\t',
    36 : ' ',
    37 : '',
    38 : '-',
    39 : '-',
}

alphaToNum = {
    1570 : 0,
    1575 : 1,
    1576 : 2,
    1662 : 3,
    1578 : 4,
    1579 : 5,
    1580 : 6,
    1670 : 7,
    1581 : 8,
    1582 : 9,
    1583 : 10,
    1584 : 11,
    1585 : 12,
    1586 : 13,
    1688 : 14,
    1587 : 15,
    1588 : 16,
    1589 : 17,
    1590 : 18,
    1591 : 19,
    1592 : 20,
    1593 : 21,
    1594 : 22,
    1601 : 23,
    1602 : 24,
    1705 : 25,
    1711 : 26,
    1604 : 27,
    1605 : 28,
    1606 : 29,
    1608 : 30,
    1607 : 31,
    1740 : 32,
    1574 : 33,
    #10 : 34, # new line
    9 : 34, # tab
    32 : 35, # space
    # 0 : 36, # nothing
    1000 : SOS_token, # start of poem
    2000 : EOS_token, # end of poem
}

numToAlpha = {v: k for k, v in alphaToNum.items()}

def convert_from_alphabet(a):
    """Encode a character
    :param a: one character
    :return: the encoded value
    """
    if a in alphaToNum:
        return alphaToNum[a]
    else:
        return None  # unknown


def convert_to_alphabet(c, avoid_tab_and_lf=False):
    """Decode a code point
    :param c: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    if c == 35:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if c == 34:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if c == 38 or c == 39: # start of poem
        return 45 # ie '-'
    if numToAlpha.has_key(c):
        return numToAlpha[c]
    else:
        return numToAlpha[0]  # unknown

def convert_to_alphabet_viz(c, avoid_tab_and_lf=False):
    """Decode a code point
    :param c: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    if c == 35:
        return ' ' if avoid_tab_and_lf else '\t'  # space instead of TAB
    if c == 34:
        return '\\' if avoid_tab_and_lf else '\n'  # \ instead of LF
    if c == 38 or c == 39: # start of poem
        return 45 # ie '-'
    if c in farsiToPingilish:
        return farsiToPingilish[c]
    else:
        return farsiToPingilish[0]  # unknown


def decode_to_text_viz(c, avoid_tab_and_lf=False):
    """Decode an encoded string.
    :param c: encoded list of code points
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return:
    """
    return "".join(map(lambda a: convert_to_alphabet_viz(a, avoid_tab_and_lf), c))


def encode_text(s):
    """Encode a string.
    :param s: a text string
    :return: encoded list of code points
    """
    output = []
    for a in s:
        ans = convert_from_alphabet(ord(a))
        if ans is not None:
            output.append(ans)

    return output#list(map(lambda a: convert_from_alphabet(ord(a)), s))


def decode_to_text(c, avoid_tab_and_lf=False):
    """Decode an encoded string.
    :param c: encoded list of code points
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return:
    """
    return "".join(map(lambda a: unichr(convert_to_alphabet(a, avoid_tab_and_lf)), c))

def readPoems(globPattern, shuffle=True):
    print("Reading poems...")

    files = glob.glob(globPattern)
    # Read the file and split into lines
    all_lines = []
    for f in files:
        lines = open(f, encoding='utf-8').readlines()
        lines = [line.strip() for line in lines]
        all_lines.extend(lines)

    if shuffle:
        random.shuffle(all_lines)
    return all_lines

def variableFromSentence(line):
    indexes = encode_text(line)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def chooseRandomWordsFromLine(line):
    words = list(set(re.split(' |\t', line)))
    random.shuffle(words)
    num_words = random.randint(MIN_WORDS, min(MAX_WORDS, len(words)))
    return '\t'.join(words[0:num_words])

def variablesFromLine(line):
    input_variable = variableFromSentence(chooseRandomWordsFromLine(line))
    target_variable = variableFromSentence(line)
    return (input_variable, target_variable)

# lines = readPoems('/home/reza/projects/ganjoorNet/ganjoor-scrapy/shahname/*.txt')
# print(variablesFromLine(lines[0]))