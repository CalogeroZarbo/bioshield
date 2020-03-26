import torch
from reformer_pytorch import ReformerLM
import argparse
import deepspeed
import json
import os
import pickle
from utils import *
from over9000 import RangerLars
import numpy as np
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for Training:', device)

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

    parser.add_argument('--path_to_file_tr', default='./chem_similarity_tr.csv', help='Trainig file') 
    parser.add_argument('--path_to_file_ts', default='./chem_similarity_ts.csv', help='Testing file') 
    parser.add_argument('--ds_conf', default='./ds_config.json', help='DeepSpeed configuration file') 
    parser.add_argument('--min_len_mol', type=int, default=-1, help='Min symbols for Canonical SMILES') 
    parser.add_argument('--max_len_mol', type=int, default=2048, help='Max symbols for Canonical SMILES.') 
    parser.add_argument('--num_examples_tr', type=int, default=1024, help='Max number of samples TR') 
    parser.add_argument('--num_examples_ts', type=int, default=1024, help='Max number of samples TS') 
    #parser.add_argument('--train_batch_size', type=int,default=8, help='Batch size') 
    parser.add_argument('--heads', type=int, default=8, help='Heads')
    parser.add_argument('--n_hashes', type=int, default=4, help='Number of hashes - 4 is permissible per author, 8 is the best but slower') 

    # parser.add_argument('--use_encdec_v2', default=False, action='store_true',
    #                     help='Use the V2 of the EncDec architecture wrapped by Philip Wang (lucidrain on github)')
    parser.add_argument('--use_full_attn', default=False, action='store_true',
                        help='Only turn on this flag to override and turn on full attention for all sequence lengths.')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args


def main():
    cmd_args = add_argument()

    path_to_file_tr = cmd_args.path_to_file_tr
    path_to_file_ts =  cmd_args.path_to_file_ts
    
    min_len_mol = cmd_args.min_len_mol
    max_len_mol = cmd_args.max_len_mol

    num_examples_tr = cmd_args.num_examples_tr
    num_examples_ts = cmd_args.num_examples_ts

    train_batch_size = json.load(open(cmd_args.ds_conf))['train_batch_size']#cmd_args.train_batch_size
    gradient_accumulation_steps = json.load(open(cmd_args.ds_conf))['gradient_accumulation_steps']
    #zero_optimization = json.load(open(cmd_args.ds_conf))['zero_optimization']
    deepspeed_optimizer = True if json.load(open(cmd_args.ds_conf)).get('optimizer', None) is not None else False

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
    use_full_attn = cmd_args.use_full_attn

    os.makedirs(output_folder, exist_ok=True)

    pickle.dump(cmd_args, open(os.sep.join([output_folder, 'training_conf.pkl']), 'wb'))

    MIN_LENGTH_MOL = min_len_mol
    MAX_LENGTH_MOL = max_len_mol # 2048
    NUM_EXAMPLES_TR = num_examples_tr # 1024
    NUM_EXAMPLES_TS = num_examples_ts # 1024
    N_EPOCHS = epochs # 10
    VALIDATE_EVERY = validate_every
    SAVE_EVERY = save_every

    MOL_SEQ_LEN = MAX_LENGTH_MOL # output_lang.max_len if (output_lang.max_len % 2) == 0  else output_lang.max_len + 1 # ??

    saved_mol_lang=os.sep.join([output_folder, 'mol_lang.pkl'])

    MAX_LENGTH_MOL = cmd_args.max_len_mol

    saved_target_lang=os.sep.join([output_folder, 'mol_lang.pkl'])
    
    mol_lang, tr_samples, ts_samples = readMolecules(genome_file_tr=path_to_file_tr, genome_file_ts=path_to_file_ts,
                                                saved_target_lang=saved_target_lang, num_examples_tr=NUM_EXAMPLES_TR, 
                                                num_examples_ts=NUM_EXAMPLES_TS, min_len_molecule= MIN_LENGTH_MOL, 
                                                max_len_molecule=MAX_LENGTH_MOL)
    
    pickle.dump(mol_lang, open(saved_mol_lang, 'wb'))

    train_dataset = MolecularSimilarityDataset(tr_samples, mol_lang, train_batch_size if device == 'cuda' else 1)
    test_dataset = MolecularSimilarityDataset(ts_samples, mol_lang, train_batch_size if device == 'cuda' else 1)

    print('Axial Embedding shape:', compute_axial_position_shape(MOL_SEQ_LEN)
        )
    model = ReformerLM(
        num_tokens = mol_lang.n_words,
        dim = dim,
        bucket_size = bucket_size,
        depth = depth, 
        heads = heads, 
        n_hashes= n_hashes,
        max_seq_len = MOL_SEQ_LEN,
        ff_chunks = ff_chunks, 
        attn_chunks = attn_chunks, 
        weight_tie = True,
        weight_tie_embedding = True,
        axial_position_emb = True,
        axial_position_shape = compute_axial_position_shape(MOL_SEQ_LEN),  
        axial_position_dims = (dim // 2, dim //2),  
        return_embeddings = True,
        use_full_attn = use_full_attn
    ).to(device)

    model_params = filter(lambda p: p.requires_grad, model.parameters())

    if deepspeed_optimizer == False:
        print('No DeepSpeed optimizer found. Using RangerLars.')
        model_optimizer = RangerLars(model.parameters()) 

        model_engine, model_optimizer, trainloader, _ = deepspeed.initialize(
            args=cmd_args, 
            model=model, 
            optimizer=model_optimizer, 
            model_parameters=model_params, 
            training_data=train_dataset
            )

    else:
        print('Found optimizer in the DeepSpeed configurations. Using it.')
        enc_dec_engine, enc_dec_optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=enc_dec, model_parameters=enc_dec_params, training_data=train_dataset)
    
    # training
    SAVE_DIR = os.sep.join([output_folder, 'saved_model'])
    os.makedirs(SAVE_DIR, exist_ok=True)
   
    try:
        enc_dec_ckp_max = np.max([int(ckp) for ckp in os.listdir(os.sep.join([SAVE_DIR,'enc_dec']))])
    except:
        enc_dec_ckp_max = 0

    _, enc_dec_client_sd = enc_dec_engine.load_checkpoint(os.sep.join([SAVE_DIR,'enc_dec']), enc_dec_ckp_max) 

    gpus_mini_batch = (train_batch_size// gradient_accumulation_steps) // torch.cuda.device_count()
    print('gpus_mini_batch:', gpus_mini_batch, 'with gradient_accumulation_steps:', gradient_accumulation_steps)
    log_file = open(os.sep.join([output_folder,'training_log.log']), 'a')
    log_file.write("\n\n\n{}\tStarting new training from chekpoint: EncoderDecoder-{}\n".format(datetime.datetime.now(), enc_dec_ckp_max))
    log_file.flush()

    testloader = model_engine.deepspeed_io(test_dataset)

    ######TO DO
    for eph in range(epochs):
        print('Starting Epoch: {}'.format(eph))
        for i, pair in enumerate(tqdm(trainloader)):
            tr_step = ((eph*len(trainloader))+i)+1

            src = pair[0]
            trg = pair[1]

            enc_dec_engine.train()
            #enc_dec.train()
            
            src = src.to(enc_dec_engine.local_rank)
            trg = trg.to(enc_dec_engine.local_rank)

            ## Need to learn how to use masks correctly
            enc_input_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in src]).bool().to(enc_dec_engine.local_rank) 
            # context_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in trg]).bool().to(device)
            #################

            loss = enc_dec_engine(src, trg, return_loss = True, enc_input_mask = enc_input_mask)#enc_input_mask)#, context_mask=context_mask)
            #loss = enc_dec(src, trg, return_loss = True, enc_input_mask = None)#enc_input_mask)#, context_mask=context_mask)

            loss.backward()

            enc_dec_engine.step()
            
            print('Training Loss:',loss.item())       
            if tr_step % validate_every == 0:
                val_loss = []
                for pair in tqdm(testloader): #Can't use the testloader or I will mess up with the model assignment and it won't learn during training, need to use normal validation instead of parallel one
                    enc_dec_engine.eval()
                    #enc_dec.eval()
                    with torch.no_grad():
                        ts_src = pair[0]
                        ts_trg = pair[1]

                        ts_src= ts_src.to(enc_dec_engine.local_rank)
                        ts_trg = ts_trg.to(enc_dec_engine.local_rank)

                        #ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
                        #ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)

                        ## Need to learn how to use masks correctly
                        ts_enc_input_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in ts_src]).bool().to(enc_dec_engine.local_rank)
                        #ts_context_mask = torch.tensor([[1 for idx in smpl if idx != PAD_IDX] for smpl in ts_trg]).bool().to(device)

                        loss = enc_dec_engine(ts_src, ts_trg, return_loss = True, enc_input_mask = ts_enc_input_mask)#ts_enc_input_mask)#, context_mask=ts_context_mask)
                        #loss = enc_dec(ts_src, ts_trg, return_loss = True, enc_input_mask = None)
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

if __name__ == "__main__":
    main()