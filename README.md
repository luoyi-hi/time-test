# TimeLagLoss

**Time Lag Loss (LagLoss)** is a new loss function for time-series forecasting that makes the model’s predictions respect the same autocorrelation patterns as the ground-truth signal.

<p align="center">
  <img src="./src/g.png" width="600">
</p>

##### Figure (g)：Comparison of model performance (Metric: MSE) with different loss guides.

我们的损失函数代码如下，是一个即插即用的模块：

```python
class TimeLagLoss(nn.Module):
    def __init__(self, args):
        super(TimeLagLoss, self).__init__()
        self.k = args.top_k
        self.loss = nn.L1Loss()
        self.alpha = args.alpha

    def diff(self, x, len):
        if len == 0:
            return x
        return x[:, len:] - x[:, :-len]

    @staticmethod
    def dedup_period(period: np.ndarray,
                     values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seen = set()
        keep_idx = []
        for i, p in enumerate(period):
            if p not in seen:
                seen.add(p)
                keep_idx.append(i)

        keep_idx = np.asarray(keep_idx, dtype=int)
        return period[keep_idx], values[keep_idx]

    def FFT_for_Period(self, x, k=2):
        xf = t.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_diff_list = frequency_list[1:] - frequency_list[:-1]

        all_top_indices, all_top_values = [], []

        _, top_list = t.topk(t.tensor(frequency_diff_list), k)
        all_top_indices.append(top_list.detach().cpu().numpy() + 1)
        all_top_values.append(frequency_list[top_list.detach().cpu().numpy() + 1])

        def safe_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        all_top_indices = [safe_numpy(idx) for idx in all_top_indices]
        all_top_values = [safe_numpy(val) for val in all_top_values]

        top_list_flat = np.concatenate(all_top_indices)
        all_top_values = np.concatenate(all_top_values)

        period = x.shape[1] // top_list_flat

        period, all_top_values = self.dedup_period(period, all_top_values)
        return period, all_top_values

    def forward(self, pred, label, input):
        period_label = label
        period_list, weight = self.FFT_for_Period(period_label, self.k)
        for i in range(len(weight)):
            if period_list[i] == 2:
                period_list = np.append(period_list, 1)
                weight = np.append(weight, weight[i])
        sm = weight.sum()
        weight /= sm
        multi_diff = 0
        pred0 = t.concat((input, pred), dim=1)
        label0 = t.concat((input, label), dim=1)
        for i in range(len(period_list)):
            w = weight[i]
            p_d = self.diff(pred0, period_list[i])
            l_d = self.diff(label0, period_list[i])
            multi_diff += self.loss(p_d, l_d) * w
        multi_diff = multi_diff + self.loss(pred.mean(dim=1, keepdim=True),
                                         label.mean(dim=1, keepdim=True)) * self.alpha
        return multi_diff
```

---



## 1. 数据集介绍

### 1.1 数据集来源
我们的数据集来源于该[github仓库]([https:](https://github.com/luoyi-hi/time-test))发布的数据集。we selected six real-world time series datasets, including ETTh1, ETTh2, ETTm1, ETTm2 which are the subsets of ETT corpus, Weather and Electricity.

- **ETT (Electricity Transformer Temperature)**: This dataset includes temperature and power load data from transformers in two regions of China, covering the years 2016 to 2018. The dataset offers two granularities: ETTh (hourly) and ETTm (15-minute intervals).
- **Weather**: This dataset captures 21 different meteorological indicators across Germany, recorded every 10 minutes throughout the year 2020. Key indicators include temperature, visibility, and other parameters, providing a comprehensive view of weather dynamics.
- **Electricity**: This dataset contains hourly electricity consumption records for 321 clients, measured in kilowatt-hours (kWh). Sourced from the UCI Machine Learning Repository, it covers the period from 2012 to 2014, offering valuable insights into consumer electricity usage patterns.

### 1.2 数据集统计信息
Dataset statistics are summarized in Table 1.
##### Table 1: Dataset statistics
<p align="center">
  <img src="./src/table1.png" width="600">
</p>

### 1.3 数据集准备
You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).  
Then place the downloaded data in the folder `./dataset`.
---



## 2. Experimental Setting
Code is based on Time Series Library (TSLib), and the datasets are also obtained from this library: https://github.com/thuml/Time-Series-Library

### 2.1 Backbone of Forecasting Models

We have selected ten representative backbone regarding time series forecasting models, encompassing diverse design principles and perspectives:

- **iTransformer**: Models different variables separately using attention mechanisms and feedforward networks to capture correlations between variables and dependencies within each variable.
- **PatchTST**: Segments time series into subseries-level patches as input tokens to Transformer and shares the same embedding and Transformer weights across all series in each channel.
- **NSTransformer**: Consists of Series Stationarization and De-stationary Attention modules to improve the predictive performance of Transformers and their variants on non-stationary time series data.
- **Autoformer**: Based on a deep decomposition architecture and self-correlation mechanism, improving long-term prediction efficiency through progressive decomposition and sequence-level connections.
- **SOFTS**: An efficient MLP-based model with a novel STAR module. Unlike traditional distributed structures, STAR uses a centralized strategy to improve efficiency and reduce reliance on channel quality.
- **Leddam**: Introduces a learnable decomposition strategy to capture dynamic trend information more reasonably and a dual attention module to capture inter-series dependencies and intra-series variations simultaneously.
- **TimeMixer**: A fully MLP-based architecture with PDM and FMM blocks to fully utilize disentangled multiscale series in both past extraction and future prediction phases.
- **DLinear**: Decomposes the time series into trend and residual sequences, and models these two sequences separately using two single-layer linear networks for prediction.
- **TSMixer**: A novel architecture designed by stacking MLPs, based on mixing operations along both the time and feature dimensions to extract information efficiently.
- **LightTS**: Compresses large ensembles into lightweight models while ensuring competitive accuracy. It proposes adaptive ensemble distillation and identifies Pareto optimal settings regarding model accuracy and size.

### 2.2 Hyperparameter Configuration.

Table 2 presents hyperparameters (batch size, learning rate, epochs, model dimensions, feed-forward dimensions, number of encoder layers) across 10 time series forecasting models on all datasets, showing dataset- and model-specific configurations for optimization. Particularly, ETT* includes the four subsets as ETTh1, ETTh2, ETTm1, ETTm2.

##### Table 2: Hyperparameter configuration.



<p align="center">
  <img src="./src/table2.png" width="800">
</p>

<p align="center">
  <img src="./src/table22.png" width="800">
</p>

---



## 3. Results Reproduction

### 3.1 Install Python

Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

### 3.2 Train and Evaluate Model

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
