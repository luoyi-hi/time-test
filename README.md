# TimeLagLoss

**Time Lag Loss (LagLoss)** is a novel, plug-and-play objective for time-series forecasting that explicitly aligns the **autocorrelation structure** of predictions with that of the ground-truth signal. By penalising errors not only at each time step but also across multiple temporal lags, LagLoss improves long-range dependency modelling and boosts forecasting accuracy on a wide range of backbones.

<p align="center">
  <img src="./src/g.png" width="600">
</p>


<p align="center"><b>Fig.&nbsp;1</b> MSE comparison of different loss functions (lower is better).</p>

---

## 1 Datasets

### 1.1 Sources  

All six datasets are taken from the public repository **[Time-Test](https://github.com/luoyi-hi/time-test)** and ultimately derived from the widely-used **[TSLib](https://github.com/thuml/Time-Series-Library)**.

| Corpus                               | Description                                                        | Span / Freq.               |
| :----------------------------------- | :----------------------------------------------------------------- | :------------------------- |
| **ETT** (ETTh1, ETTh2, ETTm1, ETTm2) | Transformer temperature & power-load data from two Chinese regions | 2016 – 2018 · 1 h / 15 min |
| **Weather**                          | 21 meteorological indicators across Germany                        | 2020 · 10 min              |
| **Electricity**                      | Hourly electricity consumption of 321 clients (UCI ML Repository)  | 2012 – 2014 · 1 h          |

### 1.2 Statistics  

Dataset statistics are summarised in **Table&nbsp;1**.

<p align="center">
  <img src="./src/table1.png" width="600">
</p>


### 1.3 Preparation  

Pre-processed data can be downloaded from **[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)** or **[Baidu Netdisk](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)** and placed in `./dataset/`.

---

## 2 Experimental Setting

### 2.1 Backbone Models  

We benchmark LagLoss on ten representative forecasting models:

- **iTransformer** — variable-wise attention with shared parameters  
- **PatchTST** — patchifying time series as Transformer tokens  
- **NSTransformer** — stationarisation & de‑stationary attention  
- **Autoformer** — series decomposition & self-correlation  
- **SOFTS** — STAR module with centralised mixing strategy  
- **Leddam** — learnable decomposition + dual attention  
- **TimeMixer** — pure-MLP past/future mixers  
- **DLinear** — linear trend–residual decomposition  
- **TSMixer** — stacked MLP mixing along time & feature axes  
- **LightTS** — distilled lightweight ensemble

### 2.2 Hyper‑parameters  

Full training hyper‑parameters for every (model, dataset) pair are provided in **Table&nbsp;2**.

<p align="center">
  <img src="./src/table2.png" width="800">
</p>


<p align="center">
  <img src="./src/table22.png" width="800">
</p>

### 2.3 Loss Function
For the baselines, we selected six loss functions, including MSE, MAE, TILDE-Q, FreDF, TDTAlign, and PSLoss. The implementation for each loss can be found in `utils\losses.py`.

**Implementation Details.** All experiments in this study were implemented within the Time-Series-Library framework<sup>1</sup>. For all models, the look-back length was consistently set to 96. For each dataset, the prediction horizons were configured as {96, 192, 336, 720}. To maintain fairness, experiments using different loss functions on the same model were conducted with uniform hyperparameters. For loss functions that require combination with MSE, we follow the settings provided in their original papers. Specifically, for FreDF, we search over $\alpha \in \{0.25, 0.5, 0.75, 1\}$, and for PS Loss, over $\alpha \in \{1, 3, 5, 10\}$. For LagLoss, we search over $\alpha \in \{0, 0.01, 0.05, 0.1, 0.15, 0.2\}$, with one exception for PatchTST, where the search range is extended to include $\{0.3, 0.5, 1\}$.


---

## 3 Plug‑and‑Play PyTorch Implementation

```python
class TimeLagLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.top_k              # dominant lags
        self.alpha = args.alpha          # mean‑error weight
        self._point = nn.L1Loss()

    # ----------------------------------------------------------------------
    def _diff(self, x, lag):
        return x if lag == 0 else x[:, lag:] - x[:, :-lag]

    @staticmethod
    def _dedup(period, val):
        seen, keep = set(), []
        for i, p in enumerate(period):
            if p not in seen:
                seen.add(p); keep.append(i)
        keep = np.asarray(keep, int)
        return period[keep], val[keep]

    def _topk_periods(self, x):
        spec = torch.abs(torch.fft.rfft(x, dim=1)).mean(0).mean(-1)
        diff = spec[1:] - spec[:-1]
        _, idx = torch.topk(diff, self.k)
        periods = x.size(1) // (idx + 1)
        periods, vals = self._dedup(periods.cpu().numpy(),
                                    spec[idx + 1].cpu().numpy())
        weight = vals / vals.sum()
        return periods, weight

    # ----------------------------------------------------------------------
    def forward(self, pred, label, hist):
        periods, w = self._topk_periods(label)
        if 2 in periods and 1 not in periods:
            periods = np.append(periods, 1)
            w = np.append(w, w[periods == 2])
            w /= w.sum()

        pred_full  = torch.cat([hist, pred],  dim=1)
        label_full = torch.cat([hist, label], dim=1)

        lag_loss = sum(
            wi * self._point(self._diff(pred_full, p),
                             self._diff(label_full, p))
            for p, wi in zip(periods, w)
        )
        mean_loss = self._point(pred.mean(1, keepdim=True),
                                label.mean(1, keepdim=True))
        return lag_loss + self.alpha * mean_loss
```

> **Drop‑in replacement** for `nn.MSELoss` / `nn.L1Loss`.

---

## 4 Result Reproduction

### 4.1 Environment  

```bash
conda create -n ts_lagloss python=3.10 -y
conda activate ts_lagloss
pip install -r requirements.txt
```

### 4.2 Training & Evaluation  

Example scripts (full list in `./scripts/`):

```bash
# iTransformer on ETTh1, horizon 96
bash scripts/long_term_forecast/ETT_script/iTransformer.sh
```

---
