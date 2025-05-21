# TimeLagLoss (Demo)

TimeLagLoss is a loss function specifically designed for time series tasks. It effectively improves the performance of time series models trained with standard MSE Loss. \
The code for CycleNet with TimeLagLoss can be found in the `CycleNet-LagLoss/` folder.  For detailed usage instructions, please refer to the README inside that folder.


## Usage

### 1. Install Python

Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).  
Then place the downloaded data in the folder `./dataset`.

### 3. Train and Evaluate Model

We provide experiment scripts for all benchmarks under the folder `./scripts/`.  
You can reproduce the experiment results with the following commands:

```bash
# ETT
bash ./scripts/long_term_forecast/ETT_script/NSTransformer.sh
bash ./scripts/long_term_forecast/ETT_script/iTransformer.sh
bash ./scripts/long_term_forecast/ETT_script/Leddam.sh
...

# ECL
bash ./scripts/long_term_forecast/ECL_script/NSTransformer.sh
...

# Weather
bash ./scripts/long_term_forecast/Weather_script/NSTransformer.sh
...
```

## Acknowledgement

Code is based on Time Series Library (TSLib), and the datasets are also obtained from this library: https://github.com/thuml/Time-Series-Library
