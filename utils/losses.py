# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""
from typing import Tuple

import torch as t
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from numba import jit
import scipy
import pdb
import torch.nn.functional as F

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
    def _dedup(lags, val):
        seen, keep = set(), []
        for i, p in enumerate(lags):
            if p not in seen:
                seen.add(p); keep.append(i)
        keep = np.asarray(keep, int)
        return lags[keep], val[keep]

    def _topk_lags(self, x):
        spec = torch.abs(torch.fft.rfft(x, dim=1)).mean(0).mean(-1)
        diff = spec[1:] - spec[:-1]
        _, idx = torch.topk(diff, self.k)
        lags = x.size(1) // (idx + 1)
        lags, vals = self._dedup(lags.cpu().numpy(),
                                    spec[idx + 1].cpu().numpy())
        weight = vals / vals.sum()
        return lags, weight

    # ----------------------------------------------------------------------
    def forward(self, pred, label, hist):
        lags, w = self._topk_lags(label)
        pred_full  = torch.cat([hist, pred],  dim=1)
        label_full = torch.cat([hist, label], dim=1)
        lag_loss = sum(
            wi * self._point(self._diff(pred_full, p),
                             self._diff(label_full, p))
            for p, wi in zip(lags, w)
        )
        mean_loss = self._point(pred.mean(1, keepdim=True),
                                label.mean(1, keepdim=True))
        return lag_loss + self.alpha * mean_loss

class ps_loss(nn.Module):
    def __init__(self, args, model):
        super(ps_loss, self).__init__()
        self.args = args
        self.model = model
        self.ps_lambda = args.alpha
        self.patch_len_threshold = args.patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.mse = nn.MSELoss()
        if args.model == 'DLinear':
            self.head = model.Linear_Trend
        if args.model == 'Autoformer':
            self.head = model.decoder.projection
        if args.model == 'iTransformer':
            self.head = model.projection
        if args.model == 'PatchTST':
            self.head = model.model.head
        if args.model == 'TimeMixer':
            self.head = model.predict_layers[-1]
        if args.model == 'SOFTS':
            self.head = model.projection
        if args.model == 'TimeXer':
            self.head = model.head.linear
        if args.model == 'Leddam':
            self.head = model.Linear_res
        if args.model == 'CycleNet':
            self.head = model.model
        if args.model == 'LightTS':
            self.head = model.layer_3.output_proj
        if args.model == 'Nonstationary_Transformer':
            self.head = model.decoder.projection

    def create_patches(self, x, patch_len, stride):
        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        B, C, L = x.shape

        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)

        return patches

    def fouriour_based_adaptive_patching(self, true, pred):
        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2

        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch

    def patch_wise_structural_loss(self, true_patch, pred_patch):
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)

        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1,
                                         keepdim=True)

        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()

        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

        return linear_corr_loss, var_loss, mean_loss

    def ps_loss(self, true, pred):
        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)

        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)

        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)
        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss

        return ps_loss

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.head.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.head.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.head.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0
        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta = gradiant_avg.norm().detach() / var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        return aplha, beta, gamma

    def forward(self, pred, label):
        loss = self.mse(pred, label)
        ps_loss = self.ps_loss(label, pred)
        loss += ps_loss * self.ps_lambda
        return loss
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


@jit(nopython=True)
def compute_softdtw(D, gamma):
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R


@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma):
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:N + 1, 1:M + 1]


@jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x


@jit(nopython=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)


@jit(nopython=True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                          V[i - 1, j - 1],
                                          V[i - 1, j]]), gamma)
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma):  # D.shape: [batch_size, N , N]
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, N + 2, N + 2)).to(device)

        for k in range(0, batch_size):  # loop over all D in the batch
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()

        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N)).to(device)
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma)
            Hessian[k:k + 1, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma=1.0):  # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size, N, N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N + 2, N + 2)).to(dev)
        for k in range(0, batch_size):  # loop over all D in the batch
            Rk = torch.FloatTensor(compute_softdtw(D_[k, :, :], g_)).to(dev)
            R[k:k + 1, :, :] = Rk
            total_loss = total_loss + Rk[-2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N, N)).to(dev)
        for k in range(batch_size):
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k, :, :], R_[k, :, :], g_)).to(dev)
            E[k:k + 1, :, :] = Ek

        return grad_output * E, None

def dilate_loss(outputs, targets, alpha=0.5, gamma=0.001):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(outputs.device)
    for k in range(batch_size):
        Dk = pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(outputs.device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal

class TDTAlign(nn.Module):
    def __init__(self):
        super(TDTAlign, self).__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def dif(self, x, len):
        if len == 0:
            return x
        return x[:, len:] - x[:, :-len]

    def sgn(self, p, l):
        a = p * l
        r = (a < 0).int()
        b, l, c = r.shape
        return r.sum().item()/(b*l*c)

    def forward(self, pred, label, input):
        pred0 = t.concat((input[:, -1:, :], pred), dim=1)
        label0 = t.concat((input[:, -1:, :], label), dim=1)
        p_d = self.dif(pred0, 1)
        l_d = self.dif(label0, 1)
        alpha = self.sgn(p_d, l_d)
        loss1 = self.mse(pred, label)
        loss2 = self.mse(p_d, l_d)
        loss = loss1*alpha + loss2 * (1-alpha)
        return loss

def fredf_loss(pred, label):
    loss_auxi = torch.fft.fft(pred, dim=1) - torch.fft.fft(label, dim=1)
    return loss_auxi.abs().mean()


def amp_loss(outputs, targets):
    # outputs = B, T, 1 --> B, 1, T
    B, _, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim=-1)

    out_norm = torch.norm(outputs, dim=-1, keepdim=True)
    tgt_norm = torch.norm(targets, dim=-1, keepdim=True)

    # calculate normalized auto correlation
    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(T - 1):], auto_corr[..., :T]], dim=-1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    # calculate cross correlation
    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim=-1).real
    cross_corr = torch.cat([cross_corr[..., -(T - 1):], cross_corr[..., :T]], dim=-1)
    nac_out = cross_corr / (tgt_norm * out_norm)

    loss = torch.mean(torch.abs(nac_tgt - nac_out))
    return loss


def ashift_loss(outputs, targets):
    B, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim=-1)))


def phase_loss(outputs, targets):
    B, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim=-1)
    tgt_fourier = torch.fft.fft(targets, dim=-1)
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    mask = (tgt_fourier_sq > (T)).float()
    topk_indices = tgt_fourier_sq.topk(k=int(T ** 0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[..., 0] = 1.
    mask = torch.where(mask > 0, 1., 0.)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    out_fourier_sq = (torch.abs(out_fourier.real) + torch.abs(out_fourier.imag))
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    phase_loss = (torch.mean(zero_error) + torch.mean(ae)) / (T ** .5)
    return phase_loss


def tildeq_loss(outputs, targets, alpha=.5, gamma=.0, beta=.5):
    outputs = outputs.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    assert not torch.isnan(outputs).any(), "Nan value detected!"
    assert not torch.isinf(outputs).any(), "Inf value detected!"
    B, _, T = outputs.shape
    l_ashift = ashift_loss(outputs, targets)
    l_amp = amp_loss(outputs, targets)
    l_phase = phase_loss(outputs, targets)
    loss = alpha * l_ashift + (1 - alpha) * l_phase + gamma * l_amp

    assert loss == loss, "Loss Nan!"
    return loss

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
