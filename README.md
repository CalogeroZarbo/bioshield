# BioShield

BioShield is a Deep Learning system to predict the Canonical SMILE configuration of a possible anti-viral molecule
and check if there are in the market FDA approved drugs ready to use.

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

## DeepLearning Seq2Seq Model

The idea behind the model is to use NeuralMachineTranslation model to "translate" the viral genome into the target
molecule. Since our main target up to now is COVID-19 we need an Encoder-Decoder that can take up to 30k sequence length
as input. This can be achieved only by using Reformer: The Efficient Transformer. The tentative training of this architecture can be found in the file `train_model_torch.py`

## ToDO

- Reformer EnoderDecoderModel

## Credits

Optimizer by: Over9000: 
- https://github.com/mgrankin/over9000

MISH:
- https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299/315
- https://medium.com/@lessw/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f
- https://github.com/digantamisra98/Mish

Refomer:
- https://github.com/lucidrains/reformer-pytorch

