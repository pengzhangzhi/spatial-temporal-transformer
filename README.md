# Overview

The *pytorch* implementation of paper: [Spatial-temporal Transformer Network with Self-supervised Learning for Traffic Flow Prediction](https://strl2022.github.io/files/paper1.pdf).


# Quick start
After download the repo and necessary dataset, you need to first generate training data and then lunach a training.

- clone this repo
```bash
git clone https://github.com/pengzhangzhi/spatial-temporal-transformer
```
- install packgaes 
```bash
pip install -r requirements.txt 
```
- Download TaxiBJ dataset and put it in the path `spatial-temporal-transformer/data/TaxiBJ/`. You only need to download `BJ16_M32x32_T30_InOut.h5`, the rest raw files are aready in the folder `spatial-temporal-transformer/data/TaxiBJ/`.
- TaxiNYC dataset is already in the repo, you do not need to download it.
- generate training data
  - generate TaxiNYC training data: `python prepareDataNY.py -c TaxiNYC.json` .
  - generate TaxiBJ training data: `python prepareDataNY.py -c TaxiBJ.json` .
  - NOTE: make sure the config file `TaxiBJ.json` are in the `config` folder.
- lunach training: `python train.py -c TaxiBJ.json`, to train the model followed the hyper-parameters in the `TaxiBJ.json` file.

# Citation

If you would like to use the code please cite my paper.
