''' Translate input text with trained model. '''
import os
import torch
import argparse
import dill as pickle
from tqdm import tqdm
import copy
import numpy as np
import sqlite3
import json

import transformer.Constants as Constants
from torchtext.legacy.data import Dataset
from transformer.Models import Transformer
from transformer.Predictor import Predictor
from util.helpers import tokens_to_source, compilation_errors

parser = argparse.ArgumentParser(description='multifix.py')

parser.add_argument('-d', '--data_name', default='DeepFix')
parser.add_argument('-sp', '--sync_pos', action='store_true')
parser.add_argument('-wsp', '--use_with_sync_pos', action='store_true')
parser.add_argument('-o', '--output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=5)
parser.add_argument('-iteration', type=int, default=5)
parser.add_argument('-max_seq_len', type=int, default=400)
parser.add_argument('-no_cuda', action='store_true')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda

iteration = opt.iteration
test_path = 'data/deepfix_raw_data'
target_vocab_path = 'vocab/target_vocab.json'
inverse_vocab_path = 'vocab/target_vocab_reverse.json'
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

data_name = opt.data_name + '_'
if opt.sync_pos:
    if opt.use_with_sync_pos:
        data_type = 'use_with_sync_pos_'
    else:
        data_type = 'sync_pos_'
else:
    data_type = ''

def load_model(opt, data_name, data_type, device):
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
        use_with_sync_pos=model_opt.use_with_sync_pos).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model

device = torch.device('cuda' if opt.cuda else 'cpu')
predictor = Predictor(
    model=load_model(opt, data_name, data_type, device),
    beam_size=opt.beam_size,
    max_seq_len=opt.max_seq_len,
    src_pad_idx=opt.src_pad_idx,
    trg_pad_idx=opt.trg_pad_idx,
    trg_bos_idx=opt.trg_bos_idx,
    trg_eos_idx=opt.trg_eos_idx,
    insert_idx=insert_idx,
    device=device).to(device)

unk_idx = SRC.vocab.stoi[SRC.unk_token]

fig_path = 'result/'
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
fig_path = fig_path + data_name + data_type + 'res/'
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
database = fig_path + "result.db"

def get_fix(program):
    src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in program]
    if opt.sync_pos:
        pred_seq = predictor.predict_sentence_with_sync_pos(torch.LongTensor([src_seq]).to(device))
    else:
        pred_seq = predictor.predict_sentence(torch.LongTensor([src_seq]).to(device))

    pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
    pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')

    return pred_line

test_dataset = np.load(os.path.join(
    test_path, 'test_raw.npy'), allow_pickle=True).item()

tonum_data =  sum([len(test_dataset[pid]) for pid in test_dataset]) # Total number of data
print("[Info] test_{} data length : {}".format(data_type, tonum_data))

conn = sqlite3.connect(database)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS programs (
                prog_id text NOT NULL,
                user_id text NOT NULL,
                prob_id text NOT NULL,
                code text NOT NULL,
                name_dict text NOT NULL,
                name_seq text NOT NULL,
                PRIMARY KEY(prog_id)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS iterations (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                fix text NOT NULL,
                PRIMARY KEY(prog_id, iteration)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS error_messages (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message text NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')

sequences_of_programs = {}
fixes_suggested_by_network = {}

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 10:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
        else:
            source = source.replace(str(l) + " ~ ", "", 1)
    source = source.replace("  ", " ")
    return source.split()

with open(inverse_vocab_path, "r") as json_file:
    inverse_vocab = json.load(json_file)

with open(target_vocab_path, "r") as json_file:
    target_vocab = json.load(json_file)

def is_replace_edit(edit):
    return str(edit) in target_vocab['replace'].values()

def apply_edits(source, edits):
    fixed = []
    inserted = 0

    for i, edit in enumerate(edits):
        if i - inserted >= len(source):
            break
        if edit == '0':
            fixed.append(source[i - inserted])
        elif edit != '-1':
            fixed.append(inverse_vocab[edit])
            if not is_replace_edit(edits[i]):
                inserted += 1
    return fixed

for problem_id, test_programs in tqdm(test_dataset.items()):
    sequences_of_programs[problem_id] = {}
    fixes_suggested_by_network[problem_id] = {}

    entries = []

    for program, name_dict, name_sequence, user_id, program_id in test_programs:
        sequences_of_programs[problem_id][program_id] = [program]
        fixes_suggested_by_network[problem_id][program_id] = []
        entries.append(
            (program, name_dict, name_sequence, user_id, program_id,))

        c.execute("INSERT OR IGNORE INTO programs VALUES (?,?,?,?,?,?)", (program_id,
                  user_id, problem_id, program, json.dumps(name_dict), json.dumps(name_sequence)))

    for round_ in range(iteration):
        to_delete = []
        input_ = []

        for i, entry in enumerate(entries):
            _, _, _, _, program_id = entry

            if sequences_of_programs[problem_id][program_id][-1] is not None:
                tmp = sequences_of_programs[problem_id][program_id][-1]
                input_.append(remove_line_numbers(tmp))
            else:
                to_delete.append(i)

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

        assert len(input_) == len(entries)

        if len(input_) == 0:
            #print('Stopping before iteration %d (no programs remain)' % (round_ + 1))
            break

        cnt = 0
        fixes = []
        for i_program in input_:
            fix = get_fix(i_program)
            fixes.append(fix)

        to_delete = []

        # Apply fixes
        for i, entry, fix in zip(range(len(fixes)), entries, fixes):
            _, _, _, _, program_id = entry
            if sum(list(map(int, fix.split()))) == 0:
                to_delete.append(i)
            else:
                program = apply_edits(remove_line_numbers(sequences_of_programs[problem_id][program_id][-1])
                                      , fix.split())
                sequences_of_programs[problem_id][program_id].append(" ".join(program))

                c.execute("INSERT OR IGNORE INTO iterations VALUES (?,?,?,?)",
                         (program_id, round_ + 1, data_name, fix))

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

    conn.commit()

conn.commit()
conn.close()

def get_final_results(database):
    with sqlite3.connect(database) as conn:
        c = conn.cursor()

        error_counts = []

        for row in c.execute("SELECT iteration, COUNT(*) FROM error_messages GROUP BY iteration ORDER BY iteration;"):
            error_counts.append(row[1])

        query1 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 0 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query1):
            initial_errors = row[0]

        query2 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 10 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query2):
            final_errors = row[0]

        query3 = """SELECT COUNT(DISTINCT prog_id)
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        query3_2 = """SELECT DISTINCT prog_id
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query3):
            fully_fixed = row[0]

        query4 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 0 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        query5 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 10 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        original_errors = {}
        for row in c.execute(query4):
            original_errors[row[0]] = int(row[1])

        partially_fixed = {}
        unfixed = {}
        for row in c.execute(query5):
            if int(row[1]) < original_errors[row[0]]:
                partially_fixed[row[0]] = int(row[1])
            elif int(row[1]) == original_errors[row[0]]:
                unfixed[row[0]] = int(row[1])
            else:
                print(row[0], row[1], original_errors[row[0]])

        token_counts = []
        assignments = None

        for row in c.execute("SELECT COUNT(DISTINCT prob_id) FROM programs p WHERE prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"):
            assignments = int(row[0])

        for row in c.execute("SELECT code FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count <> 0;"):
            token_counts += [len(row[0].split())]

        avg_token_count = np.mean(token_counts)

        print("-------")
        print("Assignments: ", assignments)
        print("Program count: ", len(token_counts))
        print("Average token count: ", avg_token_count)
        print("Error messages: ", initial_errors)
        print("-------")
        print("Errors remaining: %d (%.1f" % (final_errors,
              final_errors/initial_errors*100) + "%)")
        print("Reduction in errors: %d (%.1f" % ((initial_errors - final_errors),
              (initial_errors - final_errors)/initial_errors*100) + "%)")
        print("Completely fixed programs: %d (%.1f" % (fully_fixed,
              fully_fixed/len(token_counts)*100) + "%)")
        print("Partially fixed programs: %d (%.1f" % (len(partially_fixed),
              len(partially_fixed)/len(token_counts)*100) + "%)")
        print("Unfixed programs: %d (%.1f" % (len(unfixed),
              len(unfixed)/len(token_counts)*100) + "%)")
        print("-------")

def do_problem(problem_id):
    global reconstruction, errors, errors_full, total_count, errors_test

    c = conn.cursor()

    reconstruction[problem_id] = {}
    errors[problem_id] = {}
    errors_full[problem_id] = {}
    errors_test[problem_id] = []
    candidate_programs = []

    for row in c.execute('SELECT user_id, prog_id, code, name_dict, name_seq FROM programs WHERE prob_id = ?', (problem_id,)):
        user_id, prog_id, initial = row[0], row[1], " ".join(remove_line_numbers(row[2]))
        name_dict = json.loads(row[3])
        name_seq = json.loads(row[4])

        candidate_programs.append(
            (user_id, prog_id, initial, name_dict, name_seq,))

    for _, prog_id, initial, name_dict, name_seq in candidate_programs:
        #fixes_suggested_by_typo_network = []
        #fixes_suggested_by_undeclared_network = []
        fixes_suggested_by_network = []

        #for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'typo\' ORDER BY iteration', (prog_id,)):
        #    fixes_suggested_by_typo_network.append(row[0])

        #for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'ids\' ORDER BY iteration', (prog_id,)):
        #    fixes_suggested_by_undeclared_network.append(row[0])

        for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? ORDER BY iteration', (prog_id,)):
            fixes_suggested_by_network.append(row[0])

        reconstruction[problem_id][prog_id] = [initial]
        temp_errors, temp_errors_full = compilation_errors(
            tokens_to_source(initial, name_dict, False), database_path)
        errors[problem_id][prog_id] = [temp_errors]
        errors_full[problem_id][prog_id] = [temp_errors_full]

        for fix in fixes_suggested_by_network:
            temp_prog = " ".join(apply_edits(
                reconstruction[problem_id][prog_id][-1].split() , fix.split()))
            temp_errors, temp_errors_full = compilation_errors(
                tokens_to_source(temp_prog, name_dict, False), database_path)

            if len(temp_errors) > len(errors[problem_id][prog_id][-1]):
                break
            else:
                reconstruction[problem_id][prog_id].append(temp_prog)
                errors[problem_id][prog_id].append(temp_errors)
                errors_full[problem_id][prog_id].append(
                    temp_errors_full)

        while len(reconstruction[problem_id][prog_id]) <= 10:
            reconstruction[problem_id][prog_id].append(
                reconstruction[problem_id][prog_id][-1])
            errors[problem_id][prog_id].append(errors[problem_id][prog_id][-1])
            errors_full[problem_id][prog_id].append(
                errors_full[problem_id][prog_id][-1])

        errors_test[problem_id].append(errors[problem_id][prog_id])

        for k, errors_t, errors_full_t in zip(range(len(errors[problem_id][prog_id])), errors[problem_id][prog_id], errors_full[problem_id][prog_id]):
            c.execute("INSERT INTO error_message_strings VALUES(?, ?, ?, ?, ?)", (
                prog_id, k, 'typo', errors_full_t.decode('utf-8', 'ignore'), len(errors_t)))

            for error_ in errors_t:
                c.execute("INSERT INTO error_messages VALUES(?, ?, ?, ?)",
                            (prog_id, k, 'typo', error_.decode('utf-8', 'ignore'),))

    count_t = len(candidate_programs)
    total_count += count_t
    conn.commit()


    c.close()

conn = sqlite3.connect(database)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS error_message_strings (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message_string text NOT NULL,
                error_message_count integer NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')

problem_ids = []

for row in c.execute('SELECT DISTINCT prob_id FROM programs'):
    problem_ids.append(row[0])

c.close()

reconstruction = {}
errors = {}
errors_full = {}
errors_test = {}

fixes_per_stage = [0] * 10

total_count = 0

start = time.time()

for problem_id in tqdm(problem_ids):
    do_problem(problem_id)

time_t = time.time() - start

conn.commit()
conn.close()

print('Total time:', time_t, 'seconds')
print('Total programs processed:', total_count)
print('Average time per program:', int(float(time_t) / float(total_count) * 1000), 'ms')

total_fixes_num = {}
errors_before = {}

for problem_id in errors_test:
    total_fixes_num[problem_id] = 0

    for j, seq in enumerate(errors_test[problem_id]):
        error_numbers = [len(x) for x in seq]
        skip = False

        for i in range(len(error_numbers) - 1):
            assert (not error_numbers[i + 1] > error_numbers[i])
            total_fixes_num[problem_id] += error_numbers[i] - \
                error_numbers[i + 1]

            if error_numbers[i] != error_numbers[i + 1]:
                fixes_per_stage[i] += error_numbers[i] - error_numbers[i + 1]

total_numerator = 0
total_denominator = 0

for problem_id in errors_test:
    total_numerator += total_fixes_num[problem_id]
    total_denominator += sum([len(x[0]) for x in errors_test[problem_id]])


print(int(float(total_numerator) * 100.0 / float(total_denominator)), '%')


for stage in range(len(fixes_per_stage)):
    print('Stage', stage, ':', fixes_per_stage[stage])

get_final_results(database)
