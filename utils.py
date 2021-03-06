import torch
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import numpy as np

PAD_IDX = 0
SOS_token = 1
EOS_token = 2
SEP_token = 3


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {
            "PAD": PAD_IDX,
            "SOS": SOS_token,
            "EOS": EOS_token,
            "SEP": SEP_token
        }
        self.word2count = {"PAD": 0, "SOS": 0, "EOS": 0, "SEP": 0}
        self.index2word = {
            PAD_IDX: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            SEP_token: "SEP"
        }
        self.n_words = 4  # Count PAD, SOS and EOS
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


class GenomeToMolDataset(Dataset):
    def __init__(self, data, src_lang, trg_lang, segmentation=1):
        super().__init__()
        self.data = data
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.segmentation = segmentation

    def __getitem__(self, index):
        pair = self.data[index]
        src = torch.tensor(indexesFromSentence(self.src_lang, pair[0]),
                           dtype=torch.long)
        trg = torch.tensor(indexesFromSentence(self.trg_lang, pair[1]),
                           dtype=torch.long)
        return src, trg

    def __len__(self):
        return int(len(self.data) / self.segmentation)


class MolecularSimilarityDataset(Dataset):
    def __init__(self, data, src_lang, segmentation=1):
        super().__init__()
        self.data = data
        self.src_lang = src_lang
        self.segmentation = segmentation

    def __getitem__(self, index):
        pair = self.data[index]
        pair[1] = int(pair[1])
        src = torch.tensor(indexesFromSentence(self.src_lang, pair[0]),
                           dtype=torch.long)
        trg = torch.tensor(
            pair[1], dtype=torch.long
        )  #torch.tensor([1,0] if pair[1]==0 else [0,1], dtype=torch.long) #
        return src, trg

    def __len__(self):
        return int(len(self.data) / self.segmentation)


def compute_axial_position_shape(seq_len):
    import math

    def highestPowerof2(n):
        res = 0
        for i in range(n, 0, -1):

            # If i is a power of 2
            if ((i & (i - 1)) == 0):

                res = i
                break

        return res

    def next_power_of_2(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()

    base_n = int(math.sqrt(seq_len))

    first_component = next_power_of_2(base_n)
    second_component = highestPowerof2(base_n)

    if (first_component * second_component) != seq_len:
        second_component = 2
        first_component = int(seq_len / second_component)

    return (first_component, second_component)


def preprocess_sentence(code, max_len):
    seq = ["SOS"]
    for char in code:
        seq.append(char)
    seq.append("EOS")
    while len(seq) < max_len:
        seq.append("PAD")
    return " ".join(seq)


def preprocess_chemicals(chem1, chem2, max_len):
    seq = []

    for char in chem1:
        seq.append(char)

    seq.append("SEP")

    for char in chem2:
        seq.append(char)

    while len(seq) < max_len * 2:
        seq.append("PAD")

    return " ".join(seq)


def preprocess_mrpc(chem1, chem2, max_len):
    seq = []

    for word in chem1.split(' '):
        seq.append(word)

    seq.append("SEP")

    for word in chem2.split(' '):
        seq.append(word)

    while len(seq) < max_len * 2:
        seq.append("PAD")

    return " ".join(seq)


def readGenomes(genome_file_tr,
                genome_file_ts,
                num_examples_tr,
                num_examples_ts,
                max_len_genome,
                min_len_genome,
                max_len_molecule,
                reverse=False,
                saved_input_lang=None,
                saved_target_lang=None,
                shuffle=False):
    print("Reading lines...")
    lang1 = "Genome Virus"
    lang2 = "Molecule SMILES"

    tr_pairs = []
    # Read the file and split into lines
    with open(genome_file_tr) as csv_file:
        csv_reader = csv.reader(csv_file,
                                delimiter=",",
                                quoting=csv.QUOTE_MINIMAL)
        header = None
        for i, row in tqdm(enumerate(csv_reader)):
            if header is None:
                header = row
                continue
            if num_examples_tr > 0:
                if len(tr_pairs) == num_examples_tr:
                    break
            gen_code = row[header.index(
                'genetic_code')]  #['genetic_code'] --> 1
            can_sml = row[header.index(
                'canonical_smiles')]  #['canonical_smiles'] --> 3

            if max_len_molecule > 0:
                if len(can_sml) > max_len_molecule:
                    continue

            if min_len_genome > 0:
                if len(gen_code) < min_len_genome:
                    continue

            if max_len_genome > 0:
                if len(gen_code
                       ) <= max_len_genome - 2:  # -2 is for SOS and EOS
                    # Split line into pairs and normalize
                    tr_pairs.append([
                        preprocess_sentence(gen_code, max_len_genome),
                        preprocess_sentence(can_sml, max_len_molecule)
                    ])
            else:
                tr_pairs.append([
                    preprocess_sentence(gen_code, max_len_genome),
                    preprocess_sentence(can_sml, max_len_molecule)
                ])

    ts_pairs = []
    # Read the file and split into lines
    with open(genome_file_ts) as csv_file:
        csv_reader = csv.reader(csv_file,
                                delimiter=",",
                                quoting=csv.QUOTE_MINIMAL)
        header = None
        for i, row in tqdm(enumerate(csv_reader)):
            if header is None:
                header = row
                continue
            if num_examples_ts > 0:
                if len(ts_pairs) == num_examples_ts:
                    break
            gen_code = row[header.index(
                'genetic_code')]  #['genetic_code'] --> 1
            can_sml = row[header.index(
                'canonical_smiles')]  #['canonical_smiles'] --> 3

            if max_len_molecule > 0:
                if len(can_sml) > max_len_molecule:
                    continue

            if min_len_genome > 0:
                if len(gen_code) < min_len_genome:
                    continue

            if max_len_genome > 0:
                if len(gen_code
                       ) <= max_len_genome - 2:  # -2 is for SOS and EOS
                    # Split line into pairs and normalize
                    ts_pairs.append([
                        preprocess_sentence(gen_code, max_len_genome),
                        preprocess_sentence(can_sml, max_len_molecule)
                    ])
            else:
                ts_pairs.append([
                    preprocess_sentence(gen_code, max_len_genome),
                    preprocess_sentence(can_sml, max_len_molecule)
                ])
    if shuffle:
        np.random.shuffle(tr_pairs)
        np.random.shuffle(ts_pairs)
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

    for i, pair in enumerate(tr_pairs):
        mol = pair[1].split(' ')
        while len(mol) < max_len_mol:
            mol.append("PAD")
        tr_pairs[i][1] = ' '.join(mol)

    for i, pair in enumerate(ts_pairs):
        mol = pair[1].split(' ')
        while len(mol) < max_len_mol:
            mol.append("PAD")
        ts_pairs[i][1] = ' '.join(mol)

    if os.path.exists(saved_input_lang) and os.path.exists(saved_target_lang):
        print('Loading saved vocabs.')
        input_lang = pickle.load(open(saved_input_lang, 'rb'))
        output_lang = pickle.load(open(saved_target_lang, 'rb'))
        print('input tokens', input_lang.n_words)
        print('target_lang', output_lang.n_words)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

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


def readMolecules(molecule_file_tr,
                  molecule_file_ts,
                  num_examples_tr,
                  num_examples_ts,
                  min_len_molecule,
                  max_len_molecule,
                  reverse=False,
                  saved_molecule_lang=None,
                  shuffle=False):
    def create_samples(molecule_file, num_examples):
        samples = []
        # Read the file and split into lines
        with open(molecule_file) as csv_file:
            csv_reader = csv.reader(csv_file,
                                    delimiter=",",
                                    quoting=csv.QUOTE_MINIMAL)
            header = None
            for i, row in tqdm(enumerate(csv_reader)):
                if header is None:
                    header = row
                    continue
                if num_examples > 0:
                    if len(samples) == num_examples:
                        break

                chem1 = row[header.index('chem1')]
                chem2 = row[header.index('chem2')]
                equal = row[header.index('equal')]

                if max_len_molecule > 0:
                    if len(chem1) > max_len_molecule or len(
                            chem2) > max_len_molecule:
                        continue

                if min_len_molecule > 0:
                    if len(chem1) < min_len_molecule or len(
                            chem2) < min_len_molecule:
                        continue

                if max_len_molecule > 0:
                    if len(chem1) <= max_len_molecule - 1 and len(
                            chem2) <= max_len_molecule - 1:  # -1 is for SEP
                        # Split line into pairs and normalize
                        samples.append([
                            preprocess_chemicals(chem1, chem2,
                                                 max_len_molecule), equal
                        ])
                else:
                    samples.append([
                        preprocess_chemicals(chem1, chem2, max_len_molecule),
                        equal
                    ])
        return samples

    print("Reading lines...")
    lang1 = "Molecule SMILES"

    tr_samples = create_samples(molecule_file=molecule_file_tr,
                                num_examples=num_examples_tr)
    ts_samples = create_samples(molecule_file=molecule_file_ts,
                                num_examples=num_examples_ts)

    if shuffle:
        np.random.shuffle(tr_samples)
        np.random.shuffle(ts_samples)

    # Reverse pairs, make Lang instances
    if reverse:
        tr_samples = [list(reversed(p)) for p in tr_samples]
        ts_samples = [list(reversed(p)) for p in ts_samples]
    print("Read %s molecule pairs training" % len(tr_samples))
    print("Trimmed to %s molecule pairs training" % len(tr_samples))
    print("Read %s molecule pairs test" % len(ts_samples))
    print("Trimmed to %s molecule pairs test" % len(ts_samples))
    print("Counting words...")
    max_len_mol = 0
    for pair in tr_samples:
        mol_pair = pair[0].split(' ')
        if len(mol_pair) > max_len_mol:
            max_len_mol = len(mol_pair)

    for pair in ts_samples:
        mol_pair = pair[0].split(' ')
        if len(mol_pair) > max_len_mol:
            max_len_mol = len(mol_pair)

    for i, pair in enumerate(tr_samples):
        mol_pair = pair[0].split(' ')
        while len(mol_pair) < max_len_mol:
            mol_pair.append("PAD")
        tr_samples[i][0] = ' '.join(mol_pair)

    for i, pair in enumerate(ts_samples):
        mol_pair = pair[0].split(' ')
        while len(mol_pair) < max_len_mol:
            mol_pair.append("PAD")
        ts_samples[i][0] = ' '.join(mol_pair)

    if os.path.exists(saved_molecule_lang):
        print('Loading saved vocab.')
        molecule_lang = pickle.load(open(saved_molecule_lang, 'rb'))
        print('Molecule tokens', molecule_lang.n_words)
    else:
        molecule_lang = Lang(lang1)

    for pair in tr_samples:
        molecule_lang.addSentence(pair[0])

    for pair in ts_samples:
        molecule_lang.addSentence(pair[0])

    print("Counted words:")
    print(molecule_lang.name, molecule_lang.n_words)
    print("Max Len:")
    print(molecule_lang.name, molecule_lang.max_len)
    print("Output MaxLen measured:", max_len_mol)

    return molecule_lang, tr_samples, ts_samples


def readMRPC(molecule_file_tr,
             molecule_file_ts,
             num_examples_tr,
             num_examples_ts,
             min_len_molecule,
             max_len_molecule,
             reverse=False,
             saved_molecule_lang=None,
             shuffle=False):
    def create_samples(molecule_file, num_examples):
        samples = []
        # Read the file and split into lines
        with open(molecule_file) as csv_file:
            csv_reader = csv.reader(csv_file,
                                    delimiter=",",
                                    quoting=csv.QUOTE_MINIMAL)
            header = None
            for i, row in tqdm(enumerate(csv_reader)):
                if header is None:
                    header = row
                    continue
                if num_examples > 0:
                    if len(samples) == num_examples:
                        break

                chem1 = row[header.index('st1')]
                chem2 = row[header.index('st2')]
                equal = row[header.index('equal')]

                if max_len_molecule > 0:
                    if len(chem1.split(' ')) > max_len_molecule or len(
                            chem2.split(' ')) > max_len_molecule:
                        continue

                if min_len_molecule > 0:
                    if len(chem1.split(' ')) < min_len_molecule or len(
                            chem2.split(' ')) < min_len_molecule:
                        continue

                if max_len_molecule > 0:
                    if len(chem1.split(' ')) <= max_len_molecule - 1 and len(
                            chem2.split(
                                ' ')) <= max_len_molecule - 1:  # -1 is for SEP
                        # Split line into pairs and normalize
                        samples.append([
                            preprocess_mrpc(chem1, chem2, max_len_molecule),
                            equal
                        ])
                else:
                    samples.append([
                        preprocess_mrpc(chem1, chem2, max_len_molecule), equal
                    ])
        return samples

    print("Reading lines...")
    lang1 = "Molecule SMILES"

    tr_samples = create_samples(molecule_file=molecule_file_tr,
                                num_examples=num_examples_tr)
    ts_samples = create_samples(molecule_file=molecule_file_ts,
                                num_examples=num_examples_ts)

    if shuffle:
        np.random.shuffle(tr_samples)
        np.random.shuffle(ts_samples)

    # Reverse pairs, make Lang instances
    if reverse:
        tr_samples = [list(reversed(p)) for p in tr_samples]
        ts_samples = [list(reversed(p)) for p in ts_samples]
    print("Read %s molecule pairs training" % len(tr_samples))
    print("Trimmed to %s molecule pairs training" % len(tr_samples))
    print("Read %s molecule pairs test" % len(ts_samples))
    print("Trimmed to %s molecule pairs test" % len(ts_samples))
    print("Counting words...")
    max_len_mol = 0
    for pair in tr_samples:
        mol_pair = pair[0].split(' ')
        if len(mol_pair) > max_len_mol:
            max_len_mol = len(mol_pair)

    for pair in ts_samples:
        mol_pair = pair[0].split(' ')
        if len(mol_pair) > max_len_mol:
            max_len_mol = len(mol_pair)

    for i, pair in enumerate(tr_samples):
        mol_pair = pair[0].split(' ')
        while len(mol_pair) < max_len_mol:
            mol_pair.append("PAD")
        tr_samples[i][0] = ' '.join(mol_pair)

    for i, pair in enumerate(ts_samples):
        mol_pair = pair[0].split(' ')
        while len(mol_pair) < max_len_mol:
            mol_pair.append("PAD")
        ts_samples[i][0] = ' '.join(mol_pair)

    if os.path.exists(saved_molecule_lang):
        print('Loading saved vocab.')
        molecule_lang = pickle.load(open(saved_molecule_lang, 'rb'))
        print('Molecule tokens', molecule_lang.n_words)
    else:
        molecule_lang = Lang(lang1)

    for pair in tr_samples:
        molecule_lang.addSentence(pair[0])

    for pair in ts_samples:
        molecule_lang.addSentence(pair[0])

    print("Counted words:")
    print(molecule_lang.name, molecule_lang.n_words)
    print("Max Len:")
    print(molecule_lang.name, molecule_lang.max_len)
    print("Output MaxLen measured:", max_len_mol)

    return molecule_lang, tr_samples, ts_samples


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
