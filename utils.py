import torch
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
import os

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

def readGenomes(genome_file_tr, genome_file_ts, num_examples_tr, num_examples_ts,max_len_genome,min_len_genome,max_len_molecule,reverse=False, saved_input_lang=None, saved_target_lang=None):
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

            if min_len_genome > 0:
                if len(gen_code) < min_len_genome:
                    continue

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

    if os.path.exists(saved_input_lang) and os.path.exists(saved_target_lang):
        print('Loading saved vocabs.')
        input_lang = pickle.load(open(saved_input_lang, 'rb'))
        output_lang = pickle.load(open(saved_target_lang, 'rb'))
        print('input tokens', input_lang.n_words)
        print('target_lang',output_lang.n_words)
    else:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)


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

class GenomeToMolDataset(Dataset):
    def __init__(self, data, src_lang, trg_lang, segmentation=1):
        super().__init__()
        self.data = data
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.segmentation = segmentation

    def __getitem__(self, index):
        pair = self.data[index]
        src = torch.tensor(indexesFromSentence(self.src_lang,pair[0]), dtype=torch.long)
        trg = torch.tensor(indexesFromSentence(self.trg_lang,pair[1]), dtype=torch.long)
        return src,trg

    def __len__(self):
        return int(len(self.data) / self.segmentation)