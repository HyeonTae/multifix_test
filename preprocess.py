''' Handling the data io '''
import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext
import csv
import pandas as pd
import transformer.Constants as Constants

__author__ = "Yu-Hsiang Huang"


def main_wo_bpe():
    '''
    Usage: python preprocess.py -data_name DeepFix -save_data multifix.pkl -share_vocab
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', required=True, default='DeepFix')
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=400)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')

    opt = parser.parse_args()
    train_path = 'data/' + opt.data_name + '/test.csv'
    val_path = 'data/' + opt.data_name + '/test.csv'
    test_path = 'data/' + opt.data_name + '/test.csv'

    def tokenize_src(text):
        #for tok in text.split():
            #print("src: {}".format(tok))
        return [tok for tok in text.split()]

    def tokenize_trg(text):
        #for tok in text.split():
            #print("trg: {}".format(tok))
        return [tok for tok in text.split()]

    SRC = torchtext.legacy.data.Field(
        tokenize=tokenize_src, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    TRG = torchtext.legacy.data.Field(
        tokenize=tokenize_trg, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    train = torchtext.legacy.data.TabularDataset(
            path=train_path, format='csv',
            fields = [('src', SRC), ('trg', TRG)])

    val = torchtext.legacy.data.TabularDataset(
            path=val_path, format='csv',
            fields = [('src', SRC), ('trg', TRG)])

    test = torchtext.legacy.data.TabularDataset(
            path=test_path, format='csv',
            fields = [('src', SRC), ('trg', TRG)])

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    if opt.share_vocab:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC.vocab.stoi.items():
            # TODO: Also update the `freq`, although it is not likely to be used.
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos
        print('[Info] Get merged vocabulary size:', len(TRG.vocab))


    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))
    print('Done..')

if __name__ == '__main__':
    main_wo_bpe()
    #main()
