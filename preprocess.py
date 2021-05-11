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
    parser.add_argument('-data_name', default='DeepFix')
    parser.add_argument('-save_data', default='DeepFix.pkl')

    parser.add_argument('-max_len', type=int, default=400)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-ks', '--keep_case', action='store_true')
    parser.add_argument('-sv', '--share_vocab', action='store_true')

    opt = parser.parse_args()
    train_path = 'data/' + opt.data_name + '/test_ori.csv'
    val_path = 'data/' + opt.data_name + '/test_ori.csv'
    test_path = 'data/' + opt.data_name + '/test_ori.csv'

    def tokenize_src(text):
        return [tok for tok in text.split()]

    def tokenize_trg(text):
        return [tok for tok in text.split()]

    SRC = torchtext.legacy.data.Field(
        tokenize=tokenize_src, lower=not opt.keep_case, fix_length=400,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    TRG = torchtext.legacy.data.Field(
        tokenize=tokenize_trg, lower=not opt.keep_case, fix_length=401,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    POS = torchtext.legacy.data.Field(sequential=True, use_vocab=False,
            pad_token=None, unk_token=None, preprocessing=torchtext.legacy.data.Pipeline(lambda x: int(x)))
    SYNC_POS = torchtext.legacy.data.Field(sequential=True, use_vocab=False,
            pad_token=None, unk_token=None, preprocessing=torchtext.legacy.data.Pipeline(lambda x: int(x)))

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    train = torchtext.legacy.data.TabularDataset(
            path=train_path, format='csv',
            fields = [('src', SRC), ('trg', TRG), ('pos', POS), ('sync_pos', SYNC_POS)])

    val = torchtext.legacy.data.TabularDataset(
            path=val_path, format='csv',
            fields = [('src', SRC), ('trg', TRG), ('pos', POS), ('sync_pos', SYNC_POS)])

    test = torchtext.legacy.data.TabularDataset(
            path=test_path, format='csv',
            fields = [('src', SRC), ('trg', TRG), ('pos', POS), ('sync_pos', SYNC_POS)])

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    '''
    POS.build_vocab(train.pos, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(POS.vocab))
    SYNC_POS.build_vocab(train.sync_pos, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(SYNC_POS.vocab))

    print(POS.vocab.stoi)
    '''

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
        'vocab': {'src': SRC, 'trg': TRG, 'pos': POS, 'sync_pos': SYNC_POS},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))
    print('Done..')

if __name__ == '__main__':
    main_wo_bpe()
    #main()
