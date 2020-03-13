import torch
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from over9000 import RangerLars
import csv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
from apex import amp
import deepspeed
from torch.utils.data import DataLoader, Dataset
import argparse
import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for Training:', device)
PAD_IDX = 0
SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_IDX: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3  # Count PAD, SOS and EOS
        self.max_len = 0

    def addSentence(self, sentence):
        if len(sentence.split(' ')) > self.max_len:
            self.max_len = len(sentence.split(' '))
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def preprocess_sentence(code, max_len):
    seq = ["SOS"]
    for char in code:
        seq.append(char)
    seq.append("EOS")
    while len(seq) < max_len:
        seq.append("PAD")
    return " ".join(seq)

def readGenomes(genome_file_tr, genome_file_ts, num_examples_tr, num_examples_ts,max_len_genome,max_len_molecule,reverse=False):
    print("Reading lines...")
    lang1 = "Genome Virus"
    lang2 = "Molecule SMILES"

    tr_pairs = []
    # Read the file and split into lines
    with open(genome_file_tr) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        header = None
        for i,row in tqdm(enumerate(csv_reader)):
            if header is None:
                header = row
                continue
            if num_examples_tr > 0:
                if len(tr_pairs) == num_examples_tr:
                    break
            gen_code = row[1]#['genetic_code']
            can_sml = row[3]#['canonical_smiles']

            if max_len_genome > 0:
                if len(gen_code) < max_len_genome:
                    # Split line into pairs and normalize
                    tr_pairs.append([preprocess_sentence(gen_code, max_len_genome), preprocess_sentence(can_sml, max_len_molecule)])
            else:
                tr_pairs.append([preprocess_sentence(gen_code, max_len_genome), preprocess_sentence(can_sml, max_len_molecule)])
    
    ts_pairs = []
    # Read the file and split into lines
    with open(genome_file_ts) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        header = None
        for i,row in tqdm(enumerate(csv_reader)):
            if header is None:
                header = row
                continue
            if num_examples_ts > 0:
                if len(ts_pairs) == num_examples_ts:
                    break
            gen_code = row[1]#['genetic_code']
            can_sml = row[3]#['canonical_smiles']

            if max_len_genome > 0:
                if len(gen_code) < max_len_genome:
                    # Split line into pairs and normalize
                    ts_pairs.append([preprocess_sentence(gen_code, max_len_genome), preprocess_sentence(can_sml, max_len_molecule)])
            else:
                ts_pairs.append([preprocess_sentence(gen_code, max_len_genome), preprocess_sentence(can_sml, max_len_molecule)])
    # Reverse pairs, make Lang instances
    if reverse:
        tr_pairs = [list(reversed(p)) for p in tr_pairs]
        ts_pairs = [list(reversed(p)) for p in ts_pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    print("Read %s sentence pairs training" % len(tr_pairs))
    print("Trimmed to %s sentence pairs training" % len(tr_pairs))
    print("Read %s sentence pairs test" % len(ts_pairs))
    print("Trimmed to %s sentence pairs test" % len(ts_pairs))
    print("Counting words...")
    max_len_mol = 0
    for pair in tr_pairs:
        mol = pair[1].split(' ')
        if len(mol) > max_len_mol:
            max_len_mol = len(mol)
    for pair in ts_pairs:
        mol = pair[1].split(' ')
        if len(mol) > max_len_mol:
            max_len_mol = len(mol)

    for i,pair in enumerate(tr_pairs):
        mol = pair[1].split(' ')   
        while len(mol) < max_len_mol:
            mol.append("PAD")
        tr_pairs[i][1] = ' '.join(mol)
    
    for i,pair in enumerate(ts_pairs):
        mol = pair[1].split(' ')   
        while len(mol) < max_len_mol:
            mol.append("PAD")
        ts_pairs[i][1] = ' '.join(mol)

    for pair in tr_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    for pair in ts_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("Max Len:")
    print(input_lang.name, input_lang.max_len)
    print(output_lang.name, output_lang.max_len)
    print("Output MaxLen measured:", max_len_mol)

    return input_lang, output_lang, tr_pairs, ts_pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--ff_chunks', type=int, default=100,
                        help='Reduce memory by chunking') # 3200
    parser.add_argument('--attn_chunks', type=int, default=1,
                        help='reduce memory by chunking attention') # 128
    parser.add_argument('--dim', type=int, default=1024,
                        help='hidden layers dimension') # 128
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='input embedding dimension') # 64
    parser.add_argument('--bucket_size', type=int, default=64,
                        help='Bucket size for hashing') # 8 
    parser.add_argument('--depth', type=int, default=12,
                        help='number of hidden layers') # 12
    parser.add_argument('--validate_every', type=int, default=10,
                        help='Frequency of validation') # 12
    parser.add_argument('--save_every', type=int, default=10,
                        help='Frequency of saving checkpoint') # 12

    parser.add_argument('--path_to_file_tr', default='./gen_to_mol_tr.csv', help='Trainig file') 
    parser.add_argument('--path_to_file_ts', default='./gen_to_mol_ts.csv', help='Testing file') 
    parser.add_argument('--max_len_gen', type=int, default=32768, help='Max nucleotides per genome') 
    parser.add_argument('--max_len_mol', type=int, default=2048, help='Max symbols for Canonical SMILES') 
    parser.add_argument('--num_examples_tr', type=int, default=1024, help='Max number of samples TR') 
    parser.add_argument('--num_examples_ts', type=int, default=1024, help='Max number of samples TS') 
    parser.add_argument('--train_batch_size', type=int,default=8, help='Batch size') 
    parser.add_argument('--heads', type=int, default=8, help='Heads')
    parser.add_argument('--n_hashes', type=int, default=4, help='Number of hashes - 4 is permissible per author, 8 is the best but slower') 

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

def main():
    cmd_args = add_argument()

    path_to_file_tr = cmd_args.path_to_file_tr
    path_to_file_ts =  cmd_args.path_to_file_ts
    max_len_gen = cmd_args.max_len_gen
    max_len_mol = cmd_args.max_len_mol
    num_examples_tr = cmd_args.num_examples_tr
    num_examples_ts = cmd_args.num_examples_ts
    train_batch_size = cmd_args.train_batch_size
    epochs = cmd_args.epochs
    emb_dim = cmd_args.emb_dim
    dim = cmd_args.dim
    bucket_size = cmd_args.bucket_size
    depth = cmd_args.depth
    heads = cmd_args.heads
    n_hashes = cmd_args.n_hashes
    ff_chunks = cmd_args.ff_chunks
    attn_chunks = cmd_args.attn_chunks
    validate_every = cmd_args.validate_every
    save_every = cmd_args.save_every

    MAX_LENGTH_GEN = max_len_gen # 32768
    MAX_LENGTH_MOL = max_len_mol # 2048
    NUM_EXAMPLES_TR = num_examples_tr # 1024
    NUM_EXAMPLES_TS = num_examples_ts # 1024
    N_EPOCHS = epochs # 10
    VALIDATE_EVERY = validate_every
    SAVE_EVERY = save_every

    VIR_SEQ_LEN = MAX_LENGTH_GEN # input_lang.max_len if (input_lang.max_len % 2) == 0  else input_lang.max_len + 1 # 32000
    MOL_SEQ_LEN = MAX_LENGTH_MOL # output_lang.max_len if (output_lang.max_len % 2) == 0  else output_lang.max_len + 1 # ??
    teacher_forcing_ratio = 0.5

    input_lang, target_lang, tr_pairs, ts_pairs = readGenomes(genome_file_tr=path_to_file_tr, genome_file_ts=path_to_file_ts, 
                                                num_examples_tr=NUM_EXAMPLES_TR, num_examples_ts=NUM_EXAMPLES_TS,
                                                max_len_genome=MAX_LENGTH_GEN, max_len_molecule=MAX_LENGTH_MOL)

    class GenomeToMolDataset(Dataset):
        def __init__(self, data, src_lang, trg_lang):
            super().__init__()
            self.data = data
            self.src_lang = src_lang
            self.trg_lang = trg_lang

        def __getitem__(self, index):
            pair = self.data[index]
            src = torch.tensor(indexesFromSentence(self.src_lang,pair[0]), dtype=torch.long)
            trg = torch.tensor(indexesFromSentence(self.trg_lang,pair[1]), dtype=torch.long)
            return src,trg

        def __len__(self):
            return len(self.data)
    
    train_dataset = GenomeToMolDataset(tr_pairs, input_lang, target_lang)
    test_dataset = GenomeToMolDataset(ts_pairs, input_lang, target_lang)

    encoder = ReformerLM(
        num_tokens = input_lang.n_words,
        dim = dim,#512,
        bucket_size = bucket_size, # 16,
        depth = depth, # 6,
        heads = heads, # 8,
        n_hashes= n_hashes,
        max_seq_len = VIR_SEQ_LEN,
        ff_chunks = ff_chunks, #400,      # number of chunks for feedforward layer, make higher if there are memory issues
        attn_chunks = attn_chunks, #16,    # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
        weight_tie = True,
        weight_tie_embedding = True,
        axial_position_emb = True,
        axial_position_shape = (256, 128),  # the shape must multiply up to the max_seq_len (256 x 128 = 32768)
        axial_position_dims = (int(dim/2), int(dim/2)),  # the dims must sum up to the model dimensions (512 + 512 = 1024)
        return_embeddings = True # return output of last attention layer
    ).cuda()

    decoder = ReformerLM(
        num_tokens = target_lang.n_words,
        dim = dim, # 512,
        bucket_size = bucket_size, #16,
        depth = depth, #6,
        heads = heads, #8,
        n_hashes= n_hashes,
        ff_chunks = ff_chunks, # 400,      # number of chunks for feedforward layer, make higher if there are memory issues
        attn_chunks = attn_chunks, # 16,    # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
        max_seq_len = MOL_SEQ_LEN,
        axial_position_emb = True,
        axial_position_shape = (64, 32),  # the shape must multiply up to the max_seq_len (64 x 32 = 2048)
        axial_position_dims = (int(dim/2), int(dim/2)),  # the dims must sum up to the model dimensions (512 + 512 = 1024)
        weight_tie = True,
        weight_tie_embedding = True,
        causal = True
    ).cuda()

    encoder_optimizer = RangerLars(encoder.parameters()) 
    decoder_optimizer = RangerLars(decoder.parameters()) 

    encoder = TrainingWrapper(encoder, ignore_index=PAD_IDX).cuda()
    decoder = TrainingWrapper(decoder, ignore_index=PAD_IDX).cuda()

    encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, decoder.parameters())

    encoder_engine, encoder_optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=encoder, optimizer=encoder_optimizer, model_parameters=encoder_params, training_data=train_dataset, dist_init_required=True)
    decoder_engine, decoder_optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=decoder, optimizer=decoder_optimizer, model_parameters=encoder_params, dist_init_required=False)
   
    # training
    SAVE_DIR = './saved_model/'

    try:
        enc_ckp_max = np.max([int(ckp) for ckp in os.listdir(SAVE_DIR+'encoder/')])
    except Exception as e:
        print('Exception:', e)
        enc_ckp_max = 0
    
    try:
        dec_ckp_max = np.max([int(ckp) for ckp in os.listdir(SAVE_DIR+'decoder/')])
    except:
        dec_ckp_max = 0

    _, encoder_client_sd = encoder_engine.load_checkpoint(SAVE_DIR+'encoder/', enc_ckp_max)
    _, decoder_client_sd = decoder_engine.load_checkpoint(SAVE_DIR+'decoder/', dec_ckp_max) #args.ckpt_id 

    gpus_mini_batch = int(train_batch_size / torch.cuda.device_count())
    print('gpus_mini_batch:', gpus_mini_batch)
    log_file = open('./training_log.log', 'a')
    log_file.write("\n\n\n{}\tStarting new training from chekpoint: Encoder-{} | Decoder-{}\n".format(datetime.datetime.now(), enc_ckp_max, dec_ckp_max))
    log_file.flush()
    for eph in range(epochs):
        print('Starting Epoch: {}'.format(eph))
        for i, pair in enumerate(trainloader):
            tr_step = ((eph*len(trainloader))+i)+1

            src = pair[0]
            trg = pair[1]
            encoder_engine.train()
            decoder_engine.train()
            src = src.to(encoder_engine.local_rank)
            trg = trg.to(decoder_engine.local_rank)

            enc_keys = encoder_engine(src)
            loss = decoder_engine(trg, keys = enc_keys, return_loss = True)   # (1, 4096, 20000)
            loss.backward()

            decoder_engine.step()
            encoder_engine.step()
            
            print('Training Loss:',loss.item())       
            if tr_step % VALIDATE_EVERY == 0:
                val_loss = []
                for pair in tqdm(test_dataset):
                    encoder.eval()
                    decoder.eval()
                    with torch.no_grad():
                        ts_src = torch.tensor(np.array([pair[0].numpy()])).cuda()
                        ts_trg = torch.tensor(np.array([pair[1].numpy()])).cuda()
                        enc_keys = encoder(ts_src)
                        loss = decoder(ts_trg, keys=enc_keys, return_loss = True)
                        val_loss.append(loss.item())

                print(f'\tValidation Loss: AVG: {np.mean(val_loss)}, MEDIAN: {np.median(val_loss)}, STD: {np.std(val_loss)} ')
                log_file.write('Step: {}\tTraining Loss:{}\t Validation LOSS: AVG: {}| MEDIAN: {}| STD: {}\n'.format(
                                                                                                i,
                                                                                                loss.item(),
                                                                                                np.mean(val_loss),
                                                                                                np.median(val_loss),
                                                                                                np.std(val_loss)))
            else:
                log_file.write('Step: {}\tTraining Loss:{}\n'.format(i,loss.item()))
            
            log_file.flush()

            if tr_step % SAVE_EVERY == 0:
                print('\tSaving Checkpoint')
                enc_ckpt_id = str(enc_ckp_max+tr_step+1) 
                dec_ckpt_id = str(dec_ckp_max+tr_step+1)
                encoder_engine.save_checkpoint(SAVE_DIR+'encoder/', enc_ckpt_id)
                decoder_engine.save_checkpoint(SAVE_DIR+'decoder/', dec_ckpt_id)
        log_file.close()

if __name__ == '__main__':
    main()