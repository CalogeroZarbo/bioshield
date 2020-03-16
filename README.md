# BioShield (Work In Progress)

BioShield is intented to be an AI system able to predict the Canonical SMILE configuration of a possible anti-viral molecule. Since testing and approving a novel drug takes time, the system will be built with the capability to check if there are in the market drugs, already approved by the FDA, that are as much similar as possibile with the predicted SMILE configuration.

## Input
The input has been taken from this article: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6367519/ where a set of anti-viral molecules
were connected with the relative viruses.
Another interesting input source is: http://crdd.osdd.net/servers/avcpred/data/26_viruses_1391.txt

The virus genomes has been taken from this source: http://virusite.org/index.php which comes from this article: https://www.researchgate.net/publication/312330730_viruSITE-integrated_database_for_viral_genomics

Since they didn't match 100% I had to manually curate the two datasets in order to normalize the nomenclature of the Virus by using these two databases:
1. https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?taxid=10239#maincontent
2. https://www.ncbi.nlm.nih.gov/genome/browse#!/viruses/
   
The latter is particularly powerful since it tracks the genomes, the date of uploading and the different virus nomenclature.

For the missing Genomes, I downloaded them by using this resource: https://www.ncbi.nlm.nih.gov/genome/86693?genome_assembly_id=757732

## Requirement
1. CUDA 10.X (with PyTorch 1.2.0 --> CUDA 10.0)
2. Miniconda (https://docs.conda.io/en/latest/miniconda.html)
3. DeepSpeed (https://github.com/microsoft/DeepSpeed)
4. PyTorch (>=1.2.0)
5. ReformerPytorch (https://github.com/lucidrains/reformer-pytorch)

## Installation
- First of all install CUDA Toolkit and CuDNN, please make sure that the version of CUDA match the compiled PyTorch version.
- Install Miniconda and run `conda create -n bioshield python=3.6`
- Execute: `conda activate bioshield`
- Then install DeepSpeed by cloning the repo: https://github.com/microsoft/DeepSpeed and executing `./install.sh`
  - It will install PyTorch 1.2.0 as of 11 Mar 2020
- Execute: `pip install -r requirements.txt`
- Exdcute: `jupyter labextension install jupyterlab-plotly`

## Data Preparation

The input dataset containing the relations between Virus and Anti-Viral molecules wasn't well aligned with the
genome nomenclatures I found in the ViralGenomes DB so by using a genome-browser I managed to curate it.
The instruction to download the curated file are in the `DatasetCreation.ipynb` notebook.

Premade Genome to SMILE files can be found here:
- Training: https://storage.googleapis.com/bioshield-bucket/bioshield/gen_to_mol_tr.csv
- Testing: https://storage.googleapis.com/bioshield-bucket/bioshield/gen_to_mol_ts.csv
- Validation: https://storage.googleapis.com/bioshield-bucket/bioshield/gen_to_mol_val.csv
- Complete Dataset: https://storage.googleapis.com/bioshield-bucket/bioshield/gen_to_mol.csv

## Reformer Model Enc-Dec for Seq2Seq Model

The idea behind the model is to use NeuralMachineTranslation model to "translate" the viral genome into the target
molecule. Since our main target up to now is COVID-19 we need an Encoder-Decoder that can take up to 30k sequence length
as input. This can be achieved only by using Reformer: The Efficient Transformer. The tentative training of this architecture can be found in the file `train_seq2seq.py`

### Training the model

After executing the first part of `DataCreation.ipynb` you would have the file `gen_to_mol_tr.csv` and `gen_to_mol_ts.csv` in the root of the project.
The model is being training using `DeepSpeed` on a workstation with the following specs:
- 8 vCPU
- 30GB RAM
- 2xP100 16GB RAM

The configurations can be found in `ds_config.json` . The trained model has 24 layers in total, 12 for the Encoder and 12 for the Decoder. The hidden layers dimension is 768 neurons, and take advantage of some tricks found in the Reformer Pytorch implementation like `Axial Embeddings` that works well with long sequences, and in order to reduce the memory impact `weight_tie` has been setted to `true` both for the layers weights as well as for the embeddings ones.
For the same memory reason, the `ff_chunks` and the `attn_chunks` options has been used in order to feed in chunks the data in the model.

The optimizer chosen was the Over9000 implementation of `RangersLars` (more info at https://github.com/mgrankin/over9000). Currently the activation function used is the default one, which is `GLUE`, but in the next training I will use `MISH` (more info at https://github.com/digantamisra98/Mish).

This first training has been performed using only the first 50000 samples of the dataset. Full training will be performed on a much more powerful machine with 4x or 8x V100. Since in this initial try we have only 2 GPUs I choose 4 samples for each batch in order to parallelize them in 2 per GPU.

The testing process has been performed for 100 samples, in this initial tryout setup, every 100 training steps. In the final run the testing will be performed on the whole dataset. This training has the purpose to understand if the setup works from an architectural point of view, and if the results could start to make sense.

In order to run the training run the following command:
- `deepspeed train_seq2seq.py --dim 768 --bucket_size 64 --depth 12 --deepspeed --deepspeed_config ds_config.json --num_examples_tr 50000 --num_examples_ts 100 --ff_chunks 200 --attn_chunks 8 --validate_every 100 --save_every 100`

The first 50K trained EncoderDecoder model can be found here:
- https://storage.googleapis.com/bioshield-bucket/bioshield/first_50k_train.zip

### Validating the model

To test the model I would need also the Molecular Similarity model, in order to check which one are the most similar molecules in the dataset that shown good anti-viral properties. Also I will perform some analysis on how much different is the predicted drug with respect to the ones in the validation set. The validation procedure would be done computing the distance between the generated molecule and the ones that have good anti-viral capabilities. 

## Transformer model for molecular similarity

The idea is to take inspiration from the Transfomer models capable to achieve good performance in the STS task and put in place a similar model by comparing SMILE configurations of different chemicals.
TIP: We might use https://github.com/gmum/MAT


## ToDO

- Reformer Seq2Seq model for NMT
  - Implement Parallel Testing for the Seq2Seq model
  - Implement MISH option
    - Add MISH and MISH Cuda to automatically switch according device
  - Implement DeepSpeed evaluation script that actually generates Canonical SMILES
- SMILE Similarity Model for molecular similarity

## Disclamier

This is a work in progress, I'll surely do some mistakes or miss something. I'm sharing all my progress in real time in order to enable anyone who want to help starting from the very last update. Please let me know any issues you find in the processes, the ideas and/or the code I wrote. Contribute if you want. The main goal of this repo is trying to help during this Covid-19 pandemy. 

## Credits

Optimizer by: Over9000: 
- https://github.com/mgrankin/over9000

MISH:
- https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299/315
- https://medium.com/@lessw/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f
- https://github.com/digantamisra98/Mish

Refomer:
- https://github.com/lucidrains/reformer-pytorch

