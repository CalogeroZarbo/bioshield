import torch
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper, top_p
from over9000 import RangerLars
import os
import numpy as np
from utils import readGenomes, GenomeToMolDataset
from tqdm import tqdm
import pickle
from utils import *
import argparse
import deepspeed
import json
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device for Training:', device)


def convert_ds_chkpt(ds_chkpt, device):
    ds_state_dict = torch.load(ds_chkpt, map_location=torch.device(device))
    torch_state_dic = {}
    for k, v in ds_state_dict['module'].items():
        k = k.replace('net.net.', '')
        torch_state_dic[k] = v
    return torch_state_dic


def test_encdec_v1(input_lang, target_lang, dim, bucket_size, depth, heads,
                   n_hashes, vir_seq_len, ff_chunks, attn_chunks, mol_seq_len,
                   cmd_args, train_dataset, test_dataset, output_folder,
                   train_batch_size, epochs, validate_every, save_every,
                   checkpoint_id, deepspeed_optimizer, use_full_attn,
                   gradient_accumulation_steps, filter_thres):
    results = {
        'generated_seq': [],
        'generated_mol': [],
        'target_mol': [],
        'input_genome': []
    }

    encoder = ReformerLM(
        num_tokens=input_lang.n_words,
        dim=dim,
        bucket_size=bucket_size,
        depth=depth,
        heads=heads,
        n_hashes=n_hashes,
        max_seq_len=vir_seq_len,
        ff_chunks=ff_chunks,
        attn_chunks=attn_chunks,
        weight_tie=True,
        weight_tie_embedding=True,
        axial_position_emb=True,
        axial_position_shape=compute_axial_position_shape(vir_seq_len),
        axial_position_dims=(dim // 2, dim // 2),
        return_embeddings=True,
        use_full_attn=use_full_attn).to(device)

    decoder = ReformerLM(
        num_tokens=target_lang.n_words,
        dim=dim,
        bucket_size=bucket_size,
        depth=depth,
        heads=heads,
        n_hashes=n_hashes,
        ff_chunks=ff_chunks,
        attn_chunks=attn_chunks,
        max_seq_len=mol_seq_len,
        axial_position_emb=True,
        axial_position_shape=compute_axial_position_shape(mol_seq_len),
        axial_position_dims=(dim // 2, dim // 2),
        weight_tie=True,
        weight_tie_embedding=True,
        causal=True,
        use_full_attn=use_full_attn).to(device)

    SAVE_DIR = os.sep.join([output_folder, 'saved_model'])

    if checkpoint_id:
        enc_ckp_max = checkpoint_id
        dec_ckp_max = checkpoint_id
    else:
        try:
            enc_ckp_max = np.max([
                int(ckp)
                for ckp in os.listdir(os.sep.join([SAVE_DIR, 'encoder']))
            ])
        except Exception as e:
            print('Exception:', e)
            enc_ckp_max = 0

        try:
            dec_ckp_max = np.max([
                int(ckp)
                for ckp in os.listdir(os.sep.join([SAVE_DIR, 'decoder']))
            ])
        except:
            dec_ckp_max = 0

    encoder = TrainingWrapper(encoder, ignore_index=PAD_IDX,
                              pad_value=PAD_IDX).to(device)
    decoder = TrainingWrapper(decoder, ignore_index=PAD_IDX,
                              pad_value=PAD_IDX).to(device)
    '''
    encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, decoder.parameters())

    if deepspeed_optimizer == False:
        print('No DeepSpeed optimizer found. Using RangerLars.')
        encoder_optimizer = RangerLars(encoder.parameters())
        decoder_optimizer = RangerLars(decoder.parameters())

        encoder_engine, encoder_optimizer, trainloader, _ = deepspeed.initialize(
            args=cmd_args,
            model=encoder,
            optimizer=encoder_optimizer,
            model_parameters=encoder_params,
            training_data=train_dataset,
            dist_init_required=True
            )

        decoder_engine, decoder_optimizer, testloader, _ = deepspeed.initialize(
            args=cmd_args,
            model=decoder,
            optimizer=decoder_optimizer,
            model_parameters=decoder_params,
            training_data=test_dataset,
            dist_init_required=False
            )
    else:
        print('Found optimizer in the DeepSpeed configurations. Using it.')
        encoder_engine, encoder_optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=encoder, model_parameters=encoder_params, training_data=train_dataset, dist_init_required=True)
        decoder_engine, decoder_optimizer, testloader, _ = deepspeed.initialize(args=cmd_args, model=decoder, model_parameters=decoder_params, training_data=test_dataset, dist_init_required=False)

    _, encoder_client_sd = encoder_engine.load_checkpoint(os.sep.join([SAVE_DIR,'encoder']), enc_ckp_max)
    _, decoder_client_sd = decoder_engine.load_checkpoint(os.sep.join([SAVE_DIR,'decoder']), dec_ckp_max)

    gpus_mini_batch = (train_batch_size// gradient_accumulation_steps) // torch.cuda.device_count()
    print('gpus_mini_batch:', gpus_mini_batch, 'with gradient_accumulation_steps:', gradient_accumulation_steps)

    for pair in tqdm(testloader):
        encoder_engine.eval()
        decoder_engine.eval()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            ts_src = pair[0]
            ts_trg = pair[1]

            input_genome = [[input_lang.index2word[gen_idx.item()] for gen_idx in smpl] for smpl in pair[0]]
            target_mol = [[target_lang.index2word[mol_idx.item()] for mol_idx in smpl] for smpl in pair[1]]

            ts_src = ts_src.to(encoder_engine.local_rank) #ts_src.to(device) #
            ts_trg = ts_trg.to(decoder_engine.local_rank) #ts_trg.to(device) #

            print('ts_src.shape', ts_src.shape)
            print('ts_src.shape', ts_trg.shape)

            enc_keys = encoder(ts_src) #encoder_engine(ts_src)
            yi = torch.tensor([[SOS_token] for _ in range(gpus_mini_batch)]).long().to(decoder_engine.local_rank) #to(device) #

            #sample = decoder_engine.generate(yi, mol_seq_len, filter_logits_fn=top_p, filter_thres=0.95, keys=enc_keys, eos_token = EOS_token)
            sample = decoder.generate(yi, mol_seq_len, filter_logits_fn=top_p, filter_thres=0.95, keys=enc_keys, eos_token = EOS_token)
            actual_mol = []
            for mol_seq in sample.cpu().numpy():
                for mol_idx in mol_seq:
                    actual_mol.append(target_lang.index2word[mol_idx])
                print('Generated Seq:', sample)
                print('Generated Mol:', actual_mol)
                print('Real Mol:', target_mol[:target_mol.index(target_lang.index2word[EOS_token])])

                results['generated_seq'].append(sample)
                results['generated_mol'].append(actual_mol)
                results['target_mol'].append(target_mol)
                results['input_genome'].append(input_genome)

    print('Saving Test Results..')
    pickle.dump(results, open(os.sep.join([output_folder,'test_results.pkl']), 'wb'))
    '''

    encoder_checkpoint = os.sep.join([
        output_folder, 'saved_model', 'encoder', enc_ckp_max,
        'mp_rank_00_model_states.pt'
    ])
    decoder_checkpoint = os.sep.join([
        output_folder, 'saved_model', 'decoder', dec_ckp_max,
        'mp_rank_00_model_states.pt'
    ])

    encoder.load_state_dict(
        torch.load(encoder_checkpoint,
                   map_location=torch.device(device))['module'])
    decoder.load_state_dict(
        torch.load(decoder_checkpoint,
                   map_location=torch.device(device))['module'])

    real_batch_size = train_batch_size // gradient_accumulation_steps
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=real_batch_size,
                             shuffle=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    encoder.to(device)
    decoder.to(device)

    for pair in tqdm(test_loader):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
            ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)

            input_genome = [
                input_lang.index2word[gen_idx.item()] for gen_idx in pair[0]
            ]
            target_mol = [
                target_lang.index2word[mol_idx.item()] for mol_idx in pair[1]
            ]

            enc_keys = encoder(ts_src)
            yi = torch.tensor([[SOS_token]]).long().to(device)

            sample = decoder.generate(yi,
                                      mol_seq_len,
                                      filter_logits_fn=top_p,
                                      filter_thres=filter_thres,
                                      keys=enc_keys,
                                      eos_token=EOS_token)
            actual_mol = []
            for mol_seq in sample.cpu().numpy():
                for mol_idx in mol_seq:
                    actual_mol.append(target_lang.index2word[mol_idx])
                print('Generated Seq:', sample)
                print('Generated Mol:', actual_mol)
                print(
                    'Real Mol:',
                    target_mol[:target_mol.index(target_lang.
                                                 index2word[EOS_token])])

                results['generated_seq'].append(sample)
                results['generated_mol'].append(actual_mol)
                results['target_mol'].append(target_mol)
                results['input_genome'].append(input_genome)

    print('Saving Test Results..')
    pickle.dump(results,
                open(os.sep.join([output_folder, 'test_results.pkl']), 'wb'))
    '''
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
    '''


def test_encdec_v2(input_lang, target_lang, dim, bucket_size, vir_seq_len,
                   depth, mol_seq_len, heads, n_hashes, ff_chunks, attn_chunks,
                   cmd_args, output_folder, train_batch_size, epochs,
                   train_dataset, test_dataset, validate_every, save_every,
                   checkpoint_id, deepspeed_optimizer, use_full_attn,
                   gradient_accumulation_steps, filter_thres):
    print('Not implemented yet.')
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Testing Transformer model for Genome to SMILE translation.'
    )
    parser.add_argument(
        '--training_folder',
        type=str,
        default='./training_output',
        help='the folder where the training output has been stored')
    parser.add_argument('--checkpoint_id',
                        type=str,
                        default='1',
                        help='the checkpoint id to restore')
    parser.add_argument('--num_examples_ts',
                        type=int,
                        default=1024,
                        help='Max number of samples TS')
    parser.add_argument(
        '--filter_thres',
        type=float,
        default=0.95,
        help='Threshold to use when filtering generated tokens.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    training_folder = args.training_folder
    checkpoint_id = args.checkpoint_id
    filter_thres = args.filter_thres

    cmd_args = pickle.load(
        open(os.sep.join([training_folder, 'training_conf.pkl']), 'rb'))

    #encoder_checkpoint = os.sep.join([training_folder, 'saved_model', 'encoder', checkpoint_id,'mp_rank_00_model_states.pt'])
    #decoder_checkpoint = './saved_model/decoder/201/mp_rank_00_model_states.pt'

    path_to_file_tr = cmd_args.path_to_file_tr
    path_to_file_ts = cmd_args.path_to_file_ts
    max_len_gen = cmd_args.max_len_gen
    min_len_gen = cmd_args.min_len_gen
    max_len_mol = cmd_args.max_len_mol
    #num_examples_tr = cmd_args.num_examples_tr
    num_examples_ts = args.num_examples_ts
    train_batch_size = json.load(open(
        cmd_args.ds_conf))['train_batch_size']  #cmd_args.train_batch_size
    gradient_accumulation_steps = json.load(open(
        cmd_args.ds_conf))['gradient_accumulation_steps']
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

    deepspeed_optimizer = True if json.load(open(cmd_args.ds_conf)).get(
        'optimizer', None) is not None else False

    MAX_LENGTH_GEN = max_len_gen  # 32768
    MIN_LENGTH_GEN = min_len_gen
    MAX_LENGTH_MOL = max_len_mol  # 2048
    NUM_EXAMPLES_TR = 1  #num_examples_tr # 1024
    NUM_EXAMPLES_TS = num_examples_ts  # 1024
    N_EPOCHS = epochs  # 10
    VALIDATE_EVERY = validate_every
    SAVE_EVERY = save_every

    VIR_SEQ_LEN = MAX_LENGTH_GEN  # input_lang.max_len if (input_lang.max_len % 2) == 0  else input_lang.max_len + 1 # 32000
    MOL_SEQ_LEN = MAX_LENGTH_MOL  # output_lang.max_len if (output_lang.max_len % 2) == 0  else output_lang.max_len + 1 # ??

    saved_input_lang = os.sep.join([output_folder, 'vir_lang.pkl'])
    saved_target_lang = os.sep.join([output_folder, 'mol_lang.pkl'])

    input_lang, target_lang, tr_pairs, ts_pairs = readGenomes(
        genome_file_tr=path_to_file_tr,
        genome_file_ts=path_to_file_ts,
        saved_input_lang=saved_input_lang,
        saved_target_lang=saved_target_lang,
        num_examples_tr=NUM_EXAMPLES_TR,
        num_examples_ts=NUM_EXAMPLES_TS,
        max_len_genome=MAX_LENGTH_GEN,
        min_len_genome=MIN_LENGTH_GEN,
        max_len_molecule=MAX_LENGTH_MOL)

    #pickle.dump(input_lang, open(saved_input_lang, 'wb'))
    #pickle.dump(target_lang, open(saved_target_lang, 'wb'))

    train_dataset = GenomeToMolDataset(
        tr_pairs, input_lang, target_lang,
        train_batch_size if device == 'cuda' else 1)
    test_dataset = GenomeToMolDataset(
        ts_pairs, input_lang, target_lang,
        train_batch_size if device == 'cuda' else 1)

    ### Prapring the improved version
    if use_encdec_v2:
        test_encdec_v2(input_lang=input_lang,
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
                       checkpoint_id=checkpoint_id,
                       deepspeed_optimizer=deepspeed_optimizer,
                       use_full_attn=use_full_attn,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       filter_thres=filter_thres)
    else:
        test_encdec_v1(input_lang=input_lang,
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
                       checkpoint_id=checkpoint_id,
                       deepspeed_optimizer=deepspeed_optimizer,
                       use_full_attn=use_full_attn,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       filter_thres=filter_thres)

    # _, encoder_client_sd = encoder_engine.load_checkpoint(os.sep.join([training_folder, 'saved_model','encoder']), checkpoint_id)
    # _, decoder_client_sd = decoder_engine.load_checkpoint(os.sep.join([training_folder, 'saved_model','decoder']), checkpoint_id)

    # #encoder_optimizer = RangerLars(encoder.parameters())
    # #decoder_optimizer = RangerLars(decoder.parameters())

    # encoder = TrainingWrapper(encoder, ignore_index=PAD_IDX, pad_value=PAD_IDX)
    # decoder = TrainingWrapper(decoder, ignore_index=PAD_IDX, pad_value=PAD_IDX)

    # encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=torch.device(device))['module'])
    # decoder.load_state_dict(torch.load(decoder_checkpoint, map_location=torch.device(device))['module'])

    # enc_params_size_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, encoder.parameters())])
    # dec_params_size_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, decoder.parameters())])

    # enc_params_size = sum([np.prod(p.size()) for p in encoder.parameters()])
    # dec_params_size = sum([np.prod(p.size()) for p in decoder.parameters()])

    # print('Total parameters:', enc_params_size+dec_params_size)
    # print('Total trainable parameters:', enc_params_size_trainable+dec_params_size_trainable)

    # # for pair in tqdm(test_dataset):
    # #     encoder.eval()
    # #     decoder.eval()
    # #     with torch.no_grad():
    # #         ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
    # #         ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)
    # #         enc_keys = encoder(ts_src)
    # #         yi = torch.tensor([[SOS_token]]).long().to(device) # assume you are sampling batch size of 2, start tokens are id of 0
    # #         sample = decoder.generate(yi, MOL_SEQ_LEN, filter_logits_fn=top_p, filter_thres=0.95, keys=enc_keys, eos_token = EOS_token) # (2, <= 1024)
    # #         actual_mol = ''
    # #         for mol_seq in sample.cpu().numpy():
    # #             for mol_idx in mol_seq:
    # #                 actual_mol += target_lang.index2word[mol_idx]
    # #             print('Generated Seq:', sample)
    # #             print('Generated Mol:', actual_mol)
    # #             print('Real Mol:', [target_lang.index2word[mol_idx] for mol_idx in pair[1]])

    # val_loss = []

    # for pair in tqdm(test_dataset):
    #     encoder.eval()
    #     decoder.eval()
    #     with torch.no_grad():
    #         ts_src = torch.tensor(np.array([pair[0].numpy()])).to(device)
    #         ts_trg = torch.tensor(np.array([pair[1].numpy()])).to(device)
    #         enc_keys = encoder(ts_src)
    #         loss = decoder(ts_trg, keys=enc_keys, return_loss = True)
    #         val_loss.append(loss.item())
    #         print('Loss:', loss.item())

    # print(f'\tValidation Loss: AVG: {np.mean(val_loss)}, MEDIAN: {np.median(val_loss)}, STD: {np.std(val_loss)} ')

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


if __name__ == "__main__":
    main()
