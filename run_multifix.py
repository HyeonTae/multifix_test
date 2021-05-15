''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.legacy.data import Dataset
from transformer.Models import Transformer
from transformer.Predictor import Predictor

def load_model(opt, device):
    data_name = opt.data_name + '_'
    if opt.sync_pos:
        if opt.use_with_sync_pos:
            data_type = 'use_with_sync_pos_' + ('cat_' if not opt.add else '')
        else:
            data_type = 'sync_pos_' + ('cat_' if not opt.add else '')
    else:
        data_type = ''
    checkpoint = torch.load('output/' + data_name + data_type + 'model.chkpt', map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout,
        sync_pos=model_opt.sync_pos,
        use_with_sync_pos=model_opt.use_with_sync_pos,
        add=opt.add).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained ' + data_name + data_type + 'model state loaded.')
    return model 

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='multifix.py')

    parser.add_argument('-d', '--data_name', default='DeepFix')
    parser.add_argument('-sp', '--sync_pos', action='store_true')
    parser.add_argument('-add', action='store_true')
    parser.add_argument('-wsp', '--use_with_sync_pos', action='store_true')
    parser.add_argument('-o', '--output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=400)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open('data/' + opt.data_name + '.pkl', 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    insert_tok = list(range(1, 422))
    insert_tok = list(map(str,insert_tok))
    insert_idx = []
    for t in TRG.vocab.stoi.keys():
        if t in insert_tok:
            insert_idx.append(TRG.vocab.stoi[t])

    device = torch.device('cuda' if opt.cuda else 'cpu')
    predictor = Predictor(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx,
        insert_idx=insert_idx,
        device=device).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]

    #input_str = "_<directive>_#include _<include>_<stdio.h> _<type>_int _<id>_2@ _<op>_[ _<number>_# _<op>_] _<op>_= _<op>_{ _<number>_# _<op>_} _<op>_; _<type>_int _<id>_3@ _<op>_; _<type>_int _<id>_6@ _<op>_( _<type>_int _<id>_5@ _<op>_) _<op>_{ _<op>_} _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ _<type>_int _<id>_4@ _<op>_, _<id>_5@ _<op>_= _<number>_# _<op>_; _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_3@ _<op>_, _<op>_& _<id>_4@ _<op>_) _<op>_; _<type>_int _<id>_1@ _<op>_, _<id>_7@ _<op>_[ _<number>_# _<op>_] _<op>_; _<keyword>_for _<op>_( _<id>_1@ _<op>_= _<number>_# _<op>_; _<id>_1@ _<op>_< _<id>_3@ _<op>_; _<id>_1@ _<op>_++ _<op>_) _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_7@ _<op>_[ _<id>_1@ _<op>_] _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_1@ _<op>_= _<number>_# _<op>_; _<id>_1@ _<op>_< _<id>_3@ _<op>_; _<id>_1@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_5@ _<op>_< _<id>_7@ _<op>_[ _<id>_1@ _<op>_] _<op>_) _<id>_5@ _<op>_= _<id>_7@ _<op>_[ _<id>_1@ _<op>_] _<op>_; _<op>_} _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_5@ _<op>_) _<op>_; _<keyword>_return _<number>_# _<op>_; _<op>_}"

    #gold = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 414 211 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    #input_str = "_<directive>_#include _<include>_<stdio.h> _<directive>_#include _<include>_<stdlib.h> _<type>_int _<id>_1@ _<op>_( _<type>_int _<id>_2@ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_2@ _<op>_== _<number>_# _<op>_) _<keyword>_return _<number>_# _<op>_; _<keyword>_else _<id>_2@ _<op>_= _<id>_2@ _<op>_- _<number>_# _<op>_; _<keyword>_return _<op>_( _<op>_( _<id>_1@ _<op>_( _<id>_2@ _<op>_) _<op>_* _<op>_( _<number>_# _<op>_* _<id>_2@ _<op>_+ _<number>_# _<op>_) _<op>_* _<op>_( _<number>_# _<op>_* _<id>_2@ _<op>_+ _<number>_# _<op>_) _<op>_) _<op>_/ _<op>_( _<op>_( _<id>_2@ _<op>_+ _<number>_# _<op>_) _<op>_* _<op>_( _<id>_2@ _<op>_+ _<number>_# _<op>_) _<op>_) _<op>_) _<op>_; _<op>_} _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ _<type>_int _<id>_3@ _<op>_, _<id>_4@ _<op>_, _<id>_5@ _<op>_, _<id>_6@ _<op>_, _<id>_2@ _<op>_, _<id>_7@ _<op>_[ _<number>_# _<op>_] _<op>_; _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_2@ _<op>_) _<op>_; _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_1@ _<op>_( _<number>_# _<op>_) _<op>_) _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_3@ _<op>_= _<number>_# _<op>_; _<id>_3@ _<op>_< _<number>_# _<op>_; _<id>_3@ _<op>_++ _<op>_) _<op>_{ _<id>_7@ _<op>_[ _<id>_3@ _<op>_] _<op>_= _<id>_1@ _<op>_( _<number>_# _<op>_) _<op>_; _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_7@ _<op>_[ _<id>_3@ _<op>_] _<op>_) _<op>_. _<op>_} _<keyword>_for _<op>_( _<id>_3@ _<op>_= _<number>_# _<op>_; _<id>_3@ _<op>_< _<op>_= _<id>_2@ _<id>_3@ _<op>_++ _<op>_) _<op>_{ _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_6@ _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_4@ _<op>_= _<number>_# _<op>_; _<id>_4@ _<op>_< _<number>_# _<op>_; _<id>_4@ _<op>_++ _<op>_) _<op>_{ _<id>_5@ _<op>_= _<id>_7@ _<op>_[ _<id>_4@ _<op>_] _<keyword>_if _<op>_( _<id>_6@ _<op>_< _<number>_# _<op>_|| _<id>_6@ _<op>_< _<id>_5@ _<op>_) _<op>_{ _<APIcall>_printf _<op>_( _<string>_ _<op>_) _<op>_; _<keyword>_break _<op>_; _<keyword>_if _<op>_( _<id>_6@ _<op>_== _<id>_5@ _<op>_) _<op>_{ _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_5@ _<op>_) _<op>_; _<keyword>_break _<op>_; _<op>_} _<op>_} _<op>_} _<keyword>_return _<number>_# _<op>_; _<op>_}"

    #gold = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 810 0 0 0 0 0 0 0 0 0 0 0 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 407 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    #input_str = "_<directive>_#include _<include>_<stdio.h> _<type>_int _<id>_1@ _<op>_[ _<number>_# _<op>_] _<op>_; _<type>_int _<id>_6@ _<op>_; _<keyword>_void _<id>_2@ _<op>_( _<type>_int _<id>_4@ _<op>_, _<type>_int _<id>_1@ _<op>_[ _<op>_] _<op>_) _<op>_{ _<id>_7@ _<op>_= _<number>_# _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_8@ _<op>_= _<number>_# _<op>_; _<id>_8@ _<op>_< _<id>_6@ _<op>_; _<id>_8@ _<op>_++ _<op>_) _<op>_{ _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_1@ _<op>_[ _<id>_8@ _<op>_] _<op>_) _<op>_; _<op>_} _<op>_} _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ _<type>_int _<id>_5@ _<op>_; _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_6@ _<op>_, _<op>_& _<id>_5@ _<op>_) _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_8@ _<op>_= _<number>_# _<op>_; _<id>_8@ _<op>_< _<id>_6@ _<op>_; _<id>_8@ _<op>_++ _<op>_) _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_1@ _<op>_[ _<id>_8@ _<op>_] _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_8@ _<op>_= _<number>_# _<op>_; _<id>_8@ _<op>_< _<id>_6@ _<op>_; _<id>_8@ _<op>_++ _<op>_) _<op>_{ _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_1@ _<op>_[ _<id>_8@ _<op>_] _<op>_) _<op>_; _<op>_} _<keyword>_for _<op>_( _<type>_int _<id>_8@ _<op>_= _<number>_# _<op>_; _<id>_8@ _<op>_< _<id>_6@ _<op>_; _<id>_8@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_1@ _<op>_[ _<id>_8@ _<op>_] _<op>_> _<id>_3@ _<op>_) _<id>_3@ _<op>_= _<id>_1@ _<op>_[ _<id>_8@ _<op>_] _<op>_; _<op>_} _<id>_2@ _<op>_( _<number>_# _<op>_, _<id>_1@ _<op>_) _<op>_; _<keyword>_return _<number>_# _<op>_; _<op>_}"

    #gold = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 414 239 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 414 238 389 414 211 414 240 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    input_str = "_<directive>_#include _<include>_<stdio.h> _<type>_int _<id>_6@ _<op>_[ _<number>_# _<op>_] _<op>_; _<type>_int _<id>_1@ _<op>_; _<type>_int _<id>_4@ _<op>_( _<type>_int _<id>_7@ _<op>_) _<op>_{ _<id>_9@ _<op>_= _<number>_# _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_5@ _<op>_= _<number>_# _<op>_; _<id>_5@ _<op>_< _<id>_1@ _<op>_; _<id>_5@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_> _<id>_9@ _<op>_&& _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_< _<op>_= _<id>_7@ _<op>_) _<id>_9@ _<op>_= _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_; _<op>_} _<keyword>_return _<id>_9@ _<op>_; _<op>_} _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ _<type>_int _<id>_2@ _<op>_; _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_1@ _<op>_, _<op>_& _<id>_2@ _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_5@ _<op>_= _<number>_# _<op>_; _<id>_5@ _<op>_< _<id>_1@ _<op>_; _<id>_5@ _<op>_++ _<op>_) _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_) _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_5@ _<op>_= _<number>_# _<op>_; _<id>_5@ _<op>_< _<id>_1@ _<op>_; _<id>_5@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_> _<id>_8@ _<op>_) _<id>_8@ _<op>_= _<id>_6@ _<op>_[ _<id>_5@ _<op>_] _<op>_; _<op>_} _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_8@ _<op>_) _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_5@ _<op>_= _<number>_# _<op>_; _<id>_5@ _<op>_< _<id>_2@ _<op>_; _<id>_5@ _<op>_++ _<op>_) _<op>_{ _<id>_3@ _<op>_= _<id>_4@ _<op>_( _<id>_8@ _<op>_) _<op>_; _<id>_8@ _<op>_= _<id>_3@ _<op>_; _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_8@ _<op>_) _<op>_; _<op>_} _<keyword>_return _<number>_# _<op>_; _<op>_}"

    gold = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 414 241 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 414 211 389 0 0 0 0 0 414 240 414 233 389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    
    src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in input_str.split()]
    if opt.sync_pos:
        pred_seq = predictor.predict_sentence_with_sync_pos(torch.LongTensor([src_seq]).to(device))
    else:
        pred_seq = predictor.predict_sentence(torch.LongTensor([src_seq]).to(device))

    pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
    pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
    print("predict: {}".format(pred_line))
    print("***gold: {}".format(gold))

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python multifix.py
    '''
    main()
