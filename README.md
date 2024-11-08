# VQShape

VQShape is a pre-trained model for time-series analysis. The model provides shape-level features as interpretable representations for time-series data. The model also contains a universal codebook of shapes that generalizes to different domains and datasets. The current checkpoints are pre-trained on the UEA multivariate time-series classification datasets [[Bagnall et al., 2018]](https://timeseriesclassification.com/). 

For more details, please refer to our paper: \
&nbsp;&nbsp;&nbsp;&nbsp;[Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification](https://openreview.net/forum?id=pwKkNSuuEs) (NeurIPS 2024)\
&nbsp;&nbsp;&nbsp;&nbsp;**Authors**: Yunshi Wen, Tengfei Ma, Tsui-Wei Weng, Lam M. Nguyen, Anak Agung Julius


> **Note**
> This repository is still under construction since we are still working on cleaning up the source code. We aim to update this repository every one or two weeks. The pre-trained checkpoints will be released soon. Future updates will include benchmarking scripts for classification, forecasting, and imputation.


## Usage

### Environment Setup:

Python version: Python 3.11.

We use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to manage training, checkpointing, and potential future distributed training.

### Data Preparation
Because of the memory configuration/limitation of our computation resources, we currently implement a lazy-loading mechanism for pre-training. To prepare the pre-training data, we first read the UEA multivariate time series and save each univariate time series into a csv file. Refer to [this notebook](notebooks/data_preparation.ipynb) for more details. The [dataset](data_provider/timeseries_loader.py) can be replaced to fit specific computational resources.

### Pre-training
Specify the codebook size and the embedding dimension. For example, for a codebook size of 64 and an embedding dimension of 512, run:
```
bash ./scripts/pretrain.sh 64 512
```
Other hyperparameters and configurations can be specified in the [bash script](scripts/pretrain.sh).

### Load the pre-trained checkpoint

The pre-trained checkpoint can be loaded efficiently using the PyTorch Lightning module. Here is an example of loading the checkpoint to a CUDA device:
```python 
from vqshape.pretrain import LitVQShape

checkpoint_path = "checkpoints/uea_dim512_codebook64/VQShape.ckpt"
lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
model = lit_model.model
```

### Use the pre-trained model

#### 1. Tokenization (extract shapes from time-series)

```python
import torch
import torch.nn.functional as F
from einops import rearrange
from vqshape.pretrain import LitVQShape

# load the pre-trained model
checkpoint_path = "checkpoints/uea_dim512_codebook64/VQShape.ckpt"
lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
model = lit_model.model

x = torch.randn(16, 5, 1000)  # 16 multivariate time-series, each with 5 channels and 1000 timesteps
x = F.interpolate(x, 512, mode='linear')  # first interpolate to 512 timesteps
x = rearrange(x, 'b c t -> (b c) t')  # transform to univariate time-series

output_dict = model(x, mode='tokenize')
```

#### 2. Classification
(Coming soon.)

#### 3. Forecasting
(Coming soon.)

#### 4. Imputation
(Coming soon.)


## Pre-trained Checkpoints
(Coming soon.)


## Citation
```
@inproceedings{
    wen2024abstracted,
    title={Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification},
    author={Yunshi Wen and Tengfei Ma and Tsui-Wei Weng and Lam M. Nguyen and Anak Agung Julius},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```

## Acknowledgement

We thank the research community for the great work on time-series analysis, the open-source codebase, and the datasets, including but not limited to:
- A part of the code is adapted from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
- The UEA and UCR teams for collecting and sharing the [time-series classification datasets](https://timeseriesclassification.com/).
