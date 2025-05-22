# CycleNet with TimeLagLoss (Demo)

This model implementation is derived from the [CycleNet](https://github.com/ACAT-SCUT/CycleNet).

## Usage

### 1. Install Python

Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Our data is sourced from the [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library).  \
Create a separate folder named ./dataset and place all the CSV files in this directory. \
Note: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### 3. Train and Evaluate Model

We provide experiment script under the folder `./scripts/`.  
You can reproduce the experiment results with the following commands:

```bash
bash ./scripts/all.sh

```

