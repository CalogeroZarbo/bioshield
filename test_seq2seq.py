import torch
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper, top_p
from over9000 import RangerLars
import os
import numpy as np
from utils import readGenomes, GenomeToMolDataset
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for Training:', device)
PAD_IDX = 0
SOS_token = 1
EOS_token = 2

def convert_ds_chkpt(ds_chkpt, device):
    ds_state_dict = torch.load(ds_chkpt, map_location=torch.device(device)) 
    torch_state_dic = {} 
    for k,v in ds_state_dict['module'].items(): 
        k = k.replace('net.net.','') 
        torch_state_dic[k] = v 
    return torch_state_dic


encoder_checkpoint = './saved_model/encoder/201/mp_rank_00_model_states.pt'
decoder_checkpoint = './saved_model/decoder/201/mp_rank_00_model_states.pt'

dim = 768
bucket_size = 64
depth = 12
heads = 8
n_hashes = 4
VIR_SEQ_LEN = 32768
ff_chunks = 200
attn_chunks = 8
MOL_SEQ_LEN = 2048

output_folder = './training_output/'

input_lang, target_lang, tr_pairs, ts_pairs = readGenomes(genome_file_tr='./gen_to_mol_tr.csv', genome_file_ts='./gen_to_mol_ts.csv', 
                                                saved_input_lang='./vir_lang.pkl', saved_target_lang='./mol_lang.pkl',
                                                num_examples_tr=50000, num_examples_ts=100,
                                                max_len_genome=VIR_SEQ_LEN, min_len_genome = -1,max_len_molecule=MOL_SEQ_LEN)

pickle.dump(input_lang, open('vir_lang.pkl', 'wb'))
pickle.dump(target_lang, open('mol_lang.pkl', 'wb'))

train_dataset = GenomeToMolDataset(tr_pairs, input_lang, target_lang)
test_dataset = GenomeToMolDataset(ts_pairs, input_lang, target_lang)

encoder = ReformerLM(
    num_tokens = 11,#input_lang.n_words,
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
).to(device)

decoder = ReformerLM(
    num_tokens = 51,#target_lang.n_words,
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
).to(device)


#encoder_optimizer = RangerLars(encoder.parameters()) 
#decoder_optimizer = RangerLars(decoder.parameters()) 

encoder = TrainingWrapper(encoder, ignore_index=PAD_IDX, pad_value=PAD_IDX)
decoder = TrainingWrapper(decoder, ignore_index=PAD_IDX, pad_value=PAD_IDX)

encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=torch.device(device))['module'])
decoder.load_state_dict(torch.load(decoder_checkpoint, map_location=torch.device(device))['module'])

enc_params_size_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, encoder.parameters())])
dec_params_size_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, decoder.parameters())])

enc_params_size = sum([np.prod(p.size()) for p in encoder.parameters()])
dec_params_size = sum([np.prod(p.size()) for p in decoder.parameters()])

print('Total parameters:', enc_params_size+dec_params_size)
print('Total trainable parameters:', enc_params_size_trainable+dec_params_size_trainable)

# for pair in tqdm(test_dataset):
#     encoder.eval()
#     decoder.eval()
#     with torch.no_grad():
#         ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
#         ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)
#         enc_keys = encoder(ts_src)
#         yi = torch.tensor([[SOS_token]]).long().to(device) # assume you are sampling batch size of 2, start tokens are id of 0
#         sample = decoder.generate(yi, MOL_SEQ_LEN, filter_logits_fn=top_p, filter_thres=0.95, keys=enc_keys, eos_token = EOS_token) # (2, <= 1024)
#         actual_mol = ''
#         for mol_seq in sample.cpu().numpy():
#             for mol_idx in mol_seq:
#                 actual_mol += target_lang.index2word[mol_idx]
#             print('Generated Seq:', sample)
#             print('Generated Mol:', actual_mol)
#             print('Real Mol:', [target_lang.index2word[mol_idx] for mol_idx in pair[1]])




val_loss = []

for pair in tqdm(test_dataset):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
        ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)
        enc_keys = encoder(ts_src)
        loss = decoder(ts_trg, keys=enc_keys, return_loss = True)
        val_loss.append(loss.item())
        print('Loss:', loss.item())

print(f'\tValidation Loss: AVG: {np.mean(val_loss)}, MEDIAN: {np.median(val_loss)}, STD: {np.std(val_loss)} ')


## ENC DEC Evaluation
## evaluate with the following

# eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
# eval_seq_out_start = torch.tensor([[0.]]).long().cuda() # assume 0 is id of start token
# samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len = EN_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
# print(samples.shape) # (1, <= 1024) decode the tokens

# encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
# decoder_params = filter(lambda p: p.requires_grad, decoder.parameters())

# encoder_engine, encoder_optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=encoder, optimizer=encoder_optimizer, model_parameters=encoder_params, dist_init_required=True)
# decoder_engine, decoder_optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=decoder, optimizer=decoder_optimizer, model_parameters=encoder_params, dist_init_required=False)

# # training
# SAVE_DIR = './saved_model/'

# try:
#     enc_ckp_max = np.max([int(ckp) for ckp in os.listdir(SAVE_DIR+'encoder/')])
# except Exception as e:
#     print('Exception:', e)
#     enc_ckp_max = 0

# try:
#     dec_ckp_max = np.max([int(ckp) for ckp in os.listdir(SAVE_DIR+'decoder/')])
# except:
#     dec_ckp_max = 0

# _, encoder_client_sd = encoder_engine.load_checkpoint(SAVE_DIR+'encoder/', enc_ckp_max)
# _, decoder_client_sd = decoder_engine.load_checkpoint(SAVE_DIR+'decoder/', dec_ckp_max) #args

# torch.save(decoder.state_dict(), f'./decoder.save.pt')
# torch.save(encoder.state_dict(), f'./encoder.save.pt')


