# VQShape

VQShape is a pre-trained model for time-series analysis. The model provides shape-level features as interpretable representations for time-series data. The model also contains a universal codebook of shapes that generalizes to different domains and datasets. The current checkpoints are pre-trained on the UEA multivariate time-series classification datasets [[Bagnall et al., 2018]](https://timeseriesclassification.com/). 



(⚠️ Note: This repository is still under construction since we are still working on cleaning up the source code. We aim to update this repository every one or two weeks. The pre-trained checkpoints will be released soon. Future updates will include benchmarking scripts for classification, forecasting, and imputation.)


## Usage

### 1. Requirements:
Install Python 3.11.

### 2. Data Preparation
(Coming soon.)

### 3. Pre-training
Specify the codebook size and the embedding dimension. For example, for a codebook size of 64 and an embedding dimension of 512, run:
```
bash ./scripts/pretrain.sh 64 512
```

### 4. Classification
(Coming soon.)

### 5. Forecasting
(Coming soon.)

### 6. Imputation
(Coming soon.)


## Pre-trained Checkpoints
(Coming soon.)
