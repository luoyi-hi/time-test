# CycleNet with TimeLagLoss (Demo)

## Usage

### 1. Install Python

Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).  \
Create a separate folder named ./dataset and place all the CSV files in this directory. \
Note: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### 3. Train and Evaluate Model

We provide experiment script under the folder `./scripts/`.  
You can reproduce the experiment results with the following commands:

```bash
bash ./scripts/all.sh

```

## Acknowledgement

Code is based on CycleNet: https://github.com/ACAT-SCUT/CycleNet
