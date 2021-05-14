'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy.data import Field, Dataset, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"

def cal_performance(pred, pred_not_flatten, gold_not_flatten, gold, opt, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, opt.trg_pad_idx, smoothing=smoothing)


    pred_not_flatten = pred_not_flatten.max(2)[1]

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(opt.trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    #print(pred_not_flatten.shape, gold_not_flatten.shape)

    gold_positive = gold.ne(opt.zero_idx).eq(gold.ne(opt.eos_idx)).masked_select(non_pad_mask).sum().item()
    pred_positive = pred.ne(opt.zero_idx).eq(gold.ne(opt.eos_idx)).masked_select(non_pad_mask).sum().item()
    c_mask = gold.ne(opt.trg_pad_idx).eq(gold.ne(opt.zero_idx)).eq(gold.ne(opt.eos_idx))
    true_positive = gold.masked_select(c_mask).eq(pred.masked_select(c_mask)).sum().item()

    non_pad_mask_not_flatten = gold_not_flatten.ne(opt.trg_pad_idx)

    n_seq_correct = (pred_not_flatten.ne(gold_not_flatten) * non_pad_mask_not_flatten).sum(1).eq(0).sum().item()
    n_seq = pred_not_flatten.shape[0]


    #print(pred_not_flatten[0], gold_not_flatten[0])

    #print(pred.eq(gold).shape, pred.eq(gold).masked_select(non_pad_mask).shape)

    return loss, n_correct, n_word, n_seq_correct, n_seq, gold_positive, pred_positive, true_positive


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

def patch_pos(pos):
    pos = pos.transpose(0, 1)
    return pos

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)

    gold_not_flatten = trg[:, 1:].contiguous()

    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)

    return trg, gold, gold_not_flatten


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct, n_seq_total, n_seq_correct_total, gp_total, pp_total, tp_total = 0, 0, 0, 0, 0, 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold, gold_not_flatten = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))
        pos = patch_pos(batch.pos)
        sync_pos = patch_pos(batch.sync_pos)

        # print(gold_not_flatten.shape, gold_not_flatten[0], sync_pos.shape, sync_pos[0])

        # forward
        optimizer.zero_grad()

        #print('Prepare data src: {}'.format(src_seq.shape))
        #print('Prepare data trg: {}'.format(trg_seq.shape))
        #print('Prepare data gold: {}'.format(gold.shape))

        #print('pos: {}'.format(pos.shape))
        #print('sync_pos: {}'.format(sync_pos.shape))

        pred, pred_not_flatten = model(src_seq, trg_seq, pos, sync_pos)

        # backward and update parameters
        loss, n_correct, n_word, n_seq_correct, n_seq, gp, pp, tp = cal_performance(
            pred, pred_not_flatten, gold_not_flatten, gold, opt, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        n_seq_total += n_seq
        n_seq_correct_total += n_seq_correct

        gp_total += gp
        pp_total += pp
        tp_total += tp

        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    seq_accuracy = n_seq_correct_total/ n_seq_total

    if gp_total == 0:
        recall = 0
    else:
        recall = tp_total / gp_total

    if pp_total == 0:
        precision = 0
    else:
        precision = tp_total / pp_total

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2.0 * ((precision * recall) / (precision + recall))

    return loss_per_word, accuracy, seq_accuracy, f1_score


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct, n_seq_total, n_seq_correct_total, gp_total, pp_total, tp_total = 0, 0, 0, 0, 0, 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold, gold_not_flatten = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))
            pos = patch_pos(batch.pos)
            sync_pos = patch_pos(batch.sync_pos)

            # forward
            pred, pred_not_flatten = model(src_seq, trg_seq, pos, sync_pos)
            loss, n_correct, n_word, n_seq_correct, n_seq, gp, pp, tp = cal_performance(
                pred, pred_not_flatten, gold_not_flatten, gold, opt, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            n_seq_total += n_seq
            n_seq_correct_total += n_seq_correct

            gp_total += gp
            pp_total += pp
            tp_total += tp

            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    seq_accuracy = n_seq_correct_total/n_seq_total

    if gp_total == 0:
        recall = 0
    else:
        recall = tp_total / gp_total

    if pp_total == 0:
        precision = 0
    else:
        precision = tp_total / pp_total

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2.0 * ((precision * recall) / (precision + recall))

    return loss_per_word, accuracy, seq_accuracy, f1_score


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''
    data_name = opt.data_name + '_'
    if opt.sync_pos:
        if opt.use_with_sync_pos:
            data_type = 'use_with_sync_pos_'
        else:
            data_type = 'sync_pos_'
    else:
        data_type = ''
    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, data_name + data_type + 'train.log')
    log_valid_file = os.path.join(opt.output_dir, data_name + data_type + 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, seq_accu, f1_score, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, seq_accuracy: {seq_accu:3.3f} %, f1_score: {f1_score:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, seq_accu=100*seq_accu, f1_score=f1_score, elapse=(time.time()-start_time)/60, lr=lr))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_seq_accu, train_f1 = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, train_seq_accu, train_f1, start, lr)

        start = time.time()
        valid_loss, valid_accu, valid_seq_accu, valid_f1 = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, valid_seq_accu, valid_f1, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = data_name + data_type + 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = data_name + data_type + 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_name', default='DeepFix')
    parser.add_argument('-sp', '--sync_pos', action='store_true')
    parser.add_argument('-wsp', '--use_with_sync_pos', action='store_true')

    parser.add_argument('-epoch', type=int, default=2000)
    parser.add_argument('-b', '--batch_size', type=int, default=256)

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-es', '--embs_share_weight', action='store_true')
    parser.add_argument('-ps', '--proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-ls', '--label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    #if all((opt.train_path, opt.val_path)):
    #    training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    training_data, validation_data = prepare_dataloaders(opt, device)

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj,
        sync_pos=opt.sync_pos,
        use_with_sync_pos=opt.use_with_sync_pos).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)

def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open('data/' + opt.data_name + '.pkl', 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]
    #opt.zero_idx = data['vocab']['src'].vocab.stoi[Constants.ZERO_WORD]
    opt.zero_idx = data['vocab']['trg'].vocab.stoi[Constants.ZERO_WORD]
    opt.eos_idx = data['vocab']['trg'].vocab.stoi[Constants.EOS_WORD]


    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg'],
            'pos':data['vocab']['pos'], 'sync_pos':data['vocab']['sync_pos']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()


