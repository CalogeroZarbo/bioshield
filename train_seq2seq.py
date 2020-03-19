import torch
try:
    from reformer_pytorch import ReformerEncDec
except:
    print('ReformerEndDec not found in current version of reformer_pytorch')
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from over9000 import RangerLars
import csv
from tqdm import tqdm
import random
import numpy as np
import time
import math
from apex import amp
import deepspeed
from torch.utils.data import DataLoader, Dataset
import argparse
import datetime
import os
from utils import *
import pickle
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for Training:', device)

def compute_axial_position_shape(seq_len):
    import math
    def highestPowerof2(n): 
        res = 0; 
        for i in range(n, 0, -1): 
            
            # If i is a power of 2 
            if ((i & (i - 1)) == 0): 
            
                res = i; 
                break; 
            
        return res; 

    def next_power_of_2(x):   
        return 1 if x == 0 else 2**(x - 1).bit_length() 
    
    base_n = int(math.sqrt(seq_len))

    first_component = next_power_of_2(base_n)
    second_component = highestPowerof2(base_n)

    if (first_component*second_component) != seq_len:
        second_component = 2
        first_component = int(seq_len/second_component)
        
    return (first_component, second_component)


def train_encdec_v1(input_lang, target_lang, dim, bucket_size, depth, heads, n_hashes, vir_seq_len, ff_chunks, attn_chunks,
                    mol_seq_len, cmd_args, train_dataset, test_dataset, output_folder, train_batch_size, epochs,
                    validate_every, save_every, zero_optimization):
    print('Axial Embedding shape:', compute_axial_position_shape(vir_seq_len)
    )
    encoder = ReformerLM(
        num_tokens = input_lang.n_words,
        dim = dim,
        bucket_size = bucket_size,
        depth = depth, 
        heads = heads, 
        n_hashes= n_hashes,
        max_seq_len = vir_seq_len,
        ff_chunks = ff_chunks, 
        attn_chunks = attn_chunks, 
        weight_tie = True,
        weight_tie_embedding = True,
        axial_position_emb = True,
        axial_position_shape = compute_axial_position_shape(vir_seq_len),  
        axial_position_dims = (int(dim/2), int(dim/2)),  
        return_embeddings = True 
    ).to(device)

    decoder = ReformerLM(
        num_tokens = target_lang.n_words,
        dim = dim, 
        bucket_size = bucket_size,
        depth = depth, 
        heads = heads, 
        n_hashes= n_hashes,
        ff_chunks = ff_chunks, 
        attn_chunks = attn_chunks, 
        max_seq_len = mol_seq_len,
        axial_position_emb = True,
        axial_position_shape = compute_axial_position_shape(mol_seq_len),  
        axial_position_dims = (int(dim/2), int(dim/2)), 
        weight_tie = True,
        weight_tie_embedding = True,
        causal = True
    ).to(device)

    encoder_optimizer = RangerLars(encoder.parameters()) 
    decoder_optimizer = RangerLars(decoder.parameters()) 
    
    #if zero_optimization:
    #    encoder_optimizer = deepspeed.pt.deepspeed_zero_optimizer.FP16_DeepSpeedZeroOptimizer(encoder_optimizer)
    #    decoder_optimizer = deepspeed.pt.deepspeed_zero_optimizer.FP16_DeepSpeedZeroOptimizer(decoder_optimizer)

    encoder = TrainingWrapper(encoder, ignore_index=PAD_IDX, pad_value=PAD_IDX).to(device)
    decoder = TrainingWrapper(decoder, ignore_index=PAD_IDX, pad_value=PAD_IDX).to(device)
    

    encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, decoder.parameters())

    encoder_engine, encoder_optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=encoder, optimizer=encoder_optimizer, model_parameters=encoder_params, training_data=train_dataset, dist_init_required=True)
    decoder_engine, decoder_optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=decoder, optimizer=decoder_optimizer, model_parameters=encoder_params, dist_init_required=False)
    
    _, _, testloader, _ = deepspeed.initialize(args=cmd_args, model=decoder, optimizer=decoder_optimizer, training_data=test_dataset)
   

    SAVE_DIR = os.sep.join([output_folder, 'saved_model'])
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        enc_ckp_max = np.max([int(ckp) for ckp in os.listdir(os.sep.join([SAVE_DIR,'encoder']))])
    except Exception as e:
        print('Exception:', e)
        enc_ckp_max = 0
    
    try:
        dec_ckp_max = np.max([int(ckp) for ckp in os.listdir(os.sep.join([SAVE_DIR,'decoder']))])
    except:
        dec_ckp_max = 0

    _, encoder_client_sd = encoder_engine.load_checkpoint(os.sep.join([SAVE_DIR,'encoder']), enc_ckp_max)
    _, decoder_client_sd = decoder_engine.load_checkpoint(os.sep.join([SAVE_DIR,'decoder']), dec_ckp_max) 

    gpus_mini_batch = int(train_batch_size / torch.cuda.device_count())
    print('gpus_mini_batch:', gpus_mini_batch)
    log_file = open(os.sep.join([output_folder,'training_log.log']), 'a')
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
            if tr_step % validate_every == 0:
                val_loss = []
                for pair in tqdm(testloader):
                    encoder_engine.eval()
                    decoder_engine.eval()
                    with torch.no_grad():
                        ts_src = pair[0]
                        ts_trg = pair[1]

                        ts_src= ts_src.to(encoder_engine.local_rank)
                        ts_trg = ts_trg.to(decoder_engine.local_rank)

                        enc_keys = encoder_engine(ts_src)
                        loss = decoder_engine(ts_trg, keys=enc_keys, return_loss = True)
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

            if tr_step % save_every == 0:
                print('\tSaving Checkpoint')
                enc_ckpt_id = str(enc_ckp_max+tr_step+1) 
                dec_ckpt_id = str(dec_ckp_max+tr_step+1)
                encoder_engine.save_checkpoint(os.sep.join([SAVE_DIR,'encoder']), enc_ckpt_id)
                decoder_engine.save_checkpoint(os.sep.join([SAVE_DIR,'decoder']), dec_ckpt_id)
                
    log_file.close()
    print('\tSaving Final Checkpoint')
    enc_ckpt_id = str(enc_ckp_max+tr_step+1) 
    dec_ckpt_id = str(dec_ckp_max+tr_step+1)
    encoder_engine.save_checkpoint(os.sep.join([SAVE_DIR,'encoder']), enc_ckpt_id)
    decoder_engine.save_checkpoint(os.sep.join([SAVE_DIR,'decoder']), dec_ckpt_id)

def train_encdec_v2(input_lang, target_lang, dim, bucket_size, vir_seq_len, depth, mol_seq_len, heads, n_hashes,
                    ff_chunks, attn_chunks, cmd_args, output_folder, train_batch_size, epochs, train_dataset, test_dataset,
                    validate_every, save_every, zero_optimization):
    enc_dec = ReformerEncDec(
        dim = dim,
        bucket_size = bucket_size,

        enc_num_tokens = input_lang.n_words,
        enc_max_seq_len = vir_seq_len,
        enc_depth = depth,
        enc_bucket_size = bucket_size, 
        return_embeddings = True,

        dec_num_tokens = target_lang.n_words,
        dec_max_seq_len = mol_seq_len,
        dec_depth = depth,
        dec_bucket_size = bucket_size,
        dec_causal = True,
        
        ignore_index = PAD_IDX,
        pad_value = PAD_IDX,        
        heads = heads, 
        n_hashes= n_hashes,
        ff_chunks = ff_chunks,
        attn_chunks = attn_chunks, 
        weight_tie = True,
        weight_tie_embedding = True,
        axial_position_emb = True,
        axial_position_shape = compute_axial_position_shape(vir_seq_len),  
        axial_position_dims = (int(dim/2), int(dim/2)),  

    ).to(device)

    enc_dec_optimizer = RangerLars(enc_dec.parameters()) 

    enc_dec_params = filter(lambda p: p.requires_grad, enc_dec.parameters())

    enc_dec_engine, enc_dec_optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=enc_dec, optimizer=enc_dec_optimizer, model_parameters=enc_dec_params, training_data=train_dataset)
    _, _, testloader, _ = deepspeed.initialize(args=cmd_args, model=enc_dec, optimizer=enc_dec_optimizer, training_data=test_dataset)
   

    # training
    SAVE_DIR = os.sep.join([output_folder, 'saved_model'])
    os.makedirs(SAVE_DIR, exist_ok=True)
   
    try:
        enc_dec_ckp_max = np.max([int(ckp) for ckp in os.listdir(os.sep.join([SAVE_DIR,'enc_dec']))])
    except:
        enc_dec_ckp_max = 0

    _, enc_dec_client_sd = enc_dec_engine.load_checkpoint(os.sep.join([SAVE_DIR,'enc_dec']), enc_dec_ckp_max) 

    gpus_mini_batch = int(train_batch_size / torch.cuda.device_count())
    print('gpus_mini_batch:', gpus_mini_batch)
    log_file = open(os.sep.join([output_folder,'training_log.log']), 'a')
    log_file.write("\n\n\n{}\tStarting new training from chekpoint: EncoderDecoder-{}\n".format(datetime.datetime.now(), enc_dec_ckp_max))
    log_file.flush()

    for eph in range(epochs):
        print('Starting Epoch: {}'.format(eph))
        for i, pair in enumerate(trainloader):
            tr_step = ((eph*len(trainloader))+i)+1

            src = pair[0]
            trg = pair[1]

            enc_dec_engine.train()
            
            src = src.to(enc_dec_engine.local_rank)
            trg = trg.to(enc_dec_engine.local_rank)

            ## Need to learn how to use masks correctly
            # enc_input_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in src]).bool().to(device) 
            # context_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in trg]).bool().to(device)
            #################

            loss = enc_dec(src, trg, return_loss = True, enc_input_mask = None)#enc_input_mask)#, context_mask=context_mask)

            loss.backward()

            enc_dec_engine.step()
            
            print('Training Loss:',loss.item())       
            if tr_step % validate_every == 0:
                val_loss = []
                for pair in tqdm(testloader):
                    enc_dec_engine.eval()
                    with torch.no_grad():
                        ts_src = pair[0]
                        ts_trg = pair[1]

                        ts_src= ts_src.to(enc_dec_engine.local_rank)
                        ts_trg = ts_trg.to(enc_dec_engine.local_rank)

                        ## Need to learn how to use masks correctly
                        #ts_enc_input_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in ts_src]).bool().to(device)
                        #ts_context_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in ts_trg]).bool().to(device)

                        loss = enc_dec(ts_src, ts_trg, return_loss = True, enc_input_mask = None)#ts_enc_input_mask)#, context_mask=ts_context_mask)
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

            if tr_step % save_every == 0:
                print('\tSaving Checkpoint')
                enc_dec_ckpt_id = str(enc_dec_ckp_max+tr_step+1)
                enc_dec_engine.save_checkpoint(os.sep.join([SAVE_DIR,'enc_dec']), enc_dec_ckpt_id)
                
    log_file.close()
    print('\tSaving Final Checkpoint')
    enc_dec_ckpt_id = str(enc_dec_ckp_max+tr_step+1)
    enc_dec_engine.save_checkpoint(os.sep.join([SAVE_DIR,'enc_dec']), enc_dec_ckpt_id)

def add_argument():
    parser=argparse.ArgumentParser(description='Train Transformer Model for Genome to SMILE translation.')

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
    
    parser.add_argument('--output_folder', type=str, default='./training_output',
                        help='Output folder where to store the training output') # 12

    parser.add_argument('--path_to_file_tr', default='./gen_to_mol_tr.csv', help='Trainig file') 
    parser.add_argument('--path_to_file_ts', default='./gen_to_mol_ts.csv', help='Testing file') 
    parser.add_argument('--ds_conf', default='./ds_config.json', help='DeepSpeed configuration file') 
    parser.add_argument('--max_len_gen', type=int, default=32768, help='Max nucleotides per genome') 
    parser.add_argument('--min_len_gen', type=int, default=-1, help='Max nucleotides per genome') 
    parser.add_argument('--max_len_mol', type=int, default=2048, help='Max symbols for Canonical SMILES') 
    parser.add_argument('--num_examples_tr', type=int, default=1024, help='Max number of samples TR') 
    parser.add_argument('--num_examples_ts', type=int, default=1024, help='Max number of samples TS') 
    #parser.add_argument('--train_batch_size', type=int,default=8, help='Batch size') 
    parser.add_argument('--heads', type=int, default=8, help='Heads')
    parser.add_argument('--n_hashes', type=int, default=4, help='Number of hashes - 4 is permissible per author, 8 is the best but slower') 

    parser.add_argument('--use_encdec_v2', default=False, action='store_true',
                        help='Use the V2 of the EncDec architecture wrapped by Philip Wang (lucidrain on github)')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

def main():
    cmd_args = add_argument()

    path_to_file_tr = cmd_args.path_to_file_tr
    path_to_file_ts =  cmd_args.path_to_file_ts
    max_len_gen = cmd_args.max_len_gen
    min_len_gen = cmd_args.min_len_gen
    max_len_mol = cmd_args.max_len_mol
    num_examples_tr = cmd_args.num_examples_tr
    num_examples_ts = cmd_args.num_examples_ts

    train_batch_size = json.load(open(cmd_args.ds_conf))['train_batch_size']#cmd_args.train_batch_size
    zero_optimization = json.load(open(cmd_args.ds_conf))['zero_optimization']

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
    output_folder = cmd_args.output_folder
    use_encdec_v2 = cmd_args.use_encdec_v2

    os.makedirs(output_folder, exist_ok=True)

    pickle.dump(cmd_args, open(os.sep.join([output_folder, 'training_conf.pkl']), 'wb'))

    MAX_LENGTH_GEN = max_len_gen # 32768
    MIN_LENGTH_GEN = min_len_gen
    MAX_LENGTH_MOL = max_len_mol # 2048
    NUM_EXAMPLES_TR = num_examples_tr # 1024
    NUM_EXAMPLES_TS = num_examples_ts # 1024
    N_EPOCHS = epochs # 10
    VALIDATE_EVERY = validate_every
    SAVE_EVERY = save_every

    VIR_SEQ_LEN = MAX_LENGTH_GEN # input_lang.max_len if (input_lang.max_len % 2) == 0  else input_lang.max_len + 1 # 32000
    MOL_SEQ_LEN = MAX_LENGTH_MOL # output_lang.max_len if (output_lang.max_len % 2) == 0  else output_lang.max_len + 1 # ??

    saved_input_lang=os.sep.join([output_folder, 'vir_lang.pkl'])
    saved_target_lang=os.sep.join([output_folder, 'mol_lang.pkl'])
    
    input_lang, target_lang, tr_pairs, ts_pairs = readGenomes(genome_file_tr=path_to_file_tr, genome_file_ts=path_to_file_ts,
                                                saved_input_lang=saved_input_lang, saved_target_lang=saved_target_lang,
                                                num_examples_tr=NUM_EXAMPLES_TR, num_examples_ts=NUM_EXAMPLES_TS,
                                                max_len_genome=MAX_LENGTH_GEN, min_len_genome = MIN_LENGTH_GEN,max_len_molecule=MAX_LENGTH_MOL)
    
    pickle.dump(input_lang, open(saved_input_lang, 'wb'))
    pickle.dump(target_lang, open(saved_target_lang, 'wb'))

    train_dataset = GenomeToMolDataset(tr_pairs, input_lang, target_lang, train_batch_size if device == 'cuda' else 1)
    test_dataset = GenomeToMolDataset(ts_pairs, input_lang, target_lang, train_batch_size if device == 'cuda' else 1)

    if use_encdec_v2:
        train_encdec_v2(
            input_lang=input_lang, 
            target_lang=target_lang, 
            dim=dim, 
            bucket_size=bucket_size, 
            vir_seq_len=VIR_SEQ_LEN, 
            depth=depth, 
            mol_seq_len=MOL_SEQ_LEN, 
            heads=heads, 
            n_hashes=n_hashes,
            ff_chunks=ff_chunks, 
            attn_chunks=attn_chunks, 
            cmd_args=cmd_args, 
            output_folder=output_folder, 
            train_batch_size=train_batch_size, 
            epochs=epochs, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset,
            validate_every=VALIDATE_EVERY, 
            save_every=SAVE_EVERY,
            zero_optimization=zero_optimization
        )
    else:
        train_encdec_v1(
            input_lang=input_lang,
            target_lang=target_lang, 
            dim=dim, 
            bucket_size=bucket_size, 
            depth=depth, 
            heads=heads, 
            n_hashes=n_hashes, 
            vir_seq_len=VIR_SEQ_LEN, 
            ff_chunks=ff_chunks, 
            attn_chunks=attn_chunks,
            mol_seq_len=MOL_SEQ_LEN, 
            cmd_args=cmd_args, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            output_folder=output_folder, 
            train_batch_size=train_batch_size,
            epochs=epochs,
            validate_every=VALIDATE_EVERY, 
            save_every=SAVE_EVERY,
            zero_optimization=zero_optimization
        )



if __name__ == '__main__':
    main()