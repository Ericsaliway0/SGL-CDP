## SGL-CDP: Spectral Graph Learning for Cancer Driver Prioritization

This repository contains the code for our project,  
**"SGL-CDP: Spectral Graph Learning for Cancer Driver Prioritization,"**.
  

![Alt text](images/__overview_framework.jpeg)


## Data Source

The dataset is obtained from the following sources:

- **[STRING database](https://string-db.org/cgi/download?sessionId=b7WYyccF6G1p)**  
- **[HIPPIE: Human Integrated Protein-Protein Interaction rEference](https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/download.php)**  
- **[ConsensusPathDB (CPDB)](http://cpdb.molgen.mpg.de/CPDB)**  

These databases provide curated and integrated protein-protein interaction (PPI) and pathway data for bioinformatics research.


## Setup and Get Started

1. Install the required dependencies:
   - `pip install -r requirements.txt`

2. Activate your Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install the necessary Python packages:
   - `pip install pandas`
   - `pip install py2neo pandas matplotlib scikit-learn`
   - `pip install tqdm`
   - `pip install seaborn`

5. Install DGL:
   - `conda install -c dglteam dgl`


6. For pretraining, please download the built gene network of BRCA, and place it in `data/processed/omics_per_cancer` directory, and run the following command:
   - [Download Gene Network](https://drive.google.com/file/d/1e9QixpFot9t1mDSXIcNF6Zv6aY0IE_mA/view?usp=sharing)
   - `python gnn_embedding/gat_embedding.py --model_type GAT --out_feats 32 --num_layers 2 --num_heads 1 --batch_size 1 --lr 0.0001 --num_epochs 200`

8. For prediction, please download the data from the built gene association graph using the link below, place it in the `data/multiomics_meth/` directory, and run the following command:
   - [Download Gene Association Data](https://drive.google.com/file/d/1l7mbTn2Nxsbc7LLLJzsT8y02scD23aWo/view?usp=sharing)
   - `python main.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --learning_rate 0.001 --num_epochs 200`


<h2>Citation</h2>

<p>
If you find this project useful for your research, please cite it using the following BibTeX entry:
</p>

<pre><code>
@misc{LiMaSSRN2026Chebyshev,
  author       = {Li, Sa and Ma, Tianle},
  title        = {Learning Interpretable Gene Representations with Adaptive Chebyshev Graph Neural Networks},
  year         = {2026},
  publisher    = {SSRN},
  doi          = {10.2139/ssrn.6382922},
  url          = {https://ssrn.com/abstract=6382922},
  note         = {Available at SSRN}
}
</code></pre>