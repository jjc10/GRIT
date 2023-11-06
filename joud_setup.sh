conda create -n grit python=3.9
conda activate grit

# please change the cuda/device version as you need

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --trusted-host download.pytorch.org
pip install torch-scatter --no-cache-dir --force-reinstall --only-binary=torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org
pip install torch-sparse --no-cache-dir --force-reinstall --only-binary=torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org
pip install torch-geometric==2.4.0 --no-cache-dir --force-reinstall --only-binary=torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org
pip install torch-cluster --no-cache-dir --force-reinstall --only-binary=torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org
# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.
## conda install openbabel fsspec rdkit -c conda-forge
pip install rdkit

pip install torchmetrics==0.9.1
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym
pip install pytorch-lightning # required by graphgym
pip install setuptools==59.5.0