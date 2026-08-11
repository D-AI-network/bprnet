# BPR-Net

Official implementation of **BPR-Net: Bin-Resolved Temporal Responses for Frequency-Aware Prototype Routing in Spatiotemporal Forecasting**.

## Components

- `BinResolvedSpectralEncoder` (`BRSE`)
- `BinPrototypeAffinity` (`BPA`)
- `FrequencyAwarePrototypeRouting` (`FAPR`)
- `BPRNet`

## Install

```bash
pip install -r requirements.txt
```

## Train and test

```bash
python train.py \
  --train-npy ./data/train.npy \
  --test-npy ./data/test.npy \
  --save-dir ./outputs \
  --channels 2 \
  --use-crop true \
  --crop-mode center \
  --crop-size 20 \
  --node-mode false \
  --epochs 300
```

## Import

```python
from bprnet import BPRNet

model = BPRNet(
    h=20,
    w=20,
    c=1,
    p=25,
    node_mode=False,
)
```
