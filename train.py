import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from bprnet import BPRNet

@dataclass
class TrainConfig:
    p_in: int = 25
    batch: int = 32
    epochs: int = 50
    lr: float = 0.0005
    wd: float = 0.001
    num_workers: int = 2
    pin_memory: bool = True
    d_model: int = 128
    periods: Tuple[int, ...] = (24, 168)
    num_frequencies: int = 13
    dropout: float = 0.1
    loss_delta: float = 0.8
    seed: int = 42
    use_periodic: bool = True
    use_context: bool = True
    residual_to_last: bool = True
    num_prototypes: int = 128
    routing_epsilon: float = 0.08
    routing_temperature: float = 1.4
    use_latent_distance: bool = True
    bpa_init_alpha: float = 0.5
    q_w: float = 0.3
    node_mode: bool = True
    use_crop: bool = False
    crop_mode: str = 'center'
    crop_size: int = 20
    crop_box: Optional[Tuple[int, int, int, int]] = None
    channels: Optional[List[int]] = None
    coords_npy: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected true or false')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standardize_to_grid_or_node4d(array):
    if array.ndim == 4:
        return array
    if array.ndim == 3:
        t, n, c = array.shape
        return array.reshape(t, 1, n, c)
    if array.ndim == 2:
        t, n = array.shape
        return array.reshape(t, 1, n, 1)
    raise ValueError(f'Unsupported shape: {array.shape}')

def is_grid4d(array):
    return array.ndim == 4 and array.shape[1] > 1 and (array.shape[2] > 1)

def crop_data(array, crop_mode, crop_size, seed=None, crop_box=None):
    if array.ndim != 4:
        raise ValueError(f'crop_data expects 4D, got {array.shape}')
    _, h_orig, w_orig, _ = array.shape
    if crop_box is not None:
        h0, h1, w0, w1 = map(int, crop_box)
        return array[:, max(0, h0):min(h_orig, h1), max(0, w0):min(w_orig, w1), :]
    if crop_mode == 'none' or crop_size > min(h_orig, w_orig):
        return array
    if crop_mode == 'center':
        sh = h_orig // 2 - crop_size // 2
        sw = w_orig // 2 - crop_size // 2
    elif crop_mode == 'top_left':
        sh, sw = (0, 0)
    elif crop_mode == 'bottom_right':
        sh, sw = (h_orig - crop_size, w_orig - crop_size)
    elif crop_mode == 'random':
        rng = np.random.default_rng(seed)
        sh = int(rng.integers(0, h_orig - crop_size + 1))
        sw = int(rng.integers(0, w_orig - crop_size + 1))
    else:
        raise ValueError(f'Unknown crop_mode: {crop_mode}')
    sh = max(0, min(sh, h_orig - crop_size))
    sw = max(0, min(sw, w_orig - crop_size))
    return array[:, sh:sh + crop_size, sw:sw + crop_size, :]

def select_channels(array, channels):
    c_orig = array.shape[-1]
    if channels is None:
        return (array, list(range(c_orig)))
    invalid = [ch for ch in channels if not isinstance(ch, (int, np.integer)) or ch < 0 or ch >= c_orig]
    if invalid:
        raise ValueError(f'Invalid channels: {invalid}')
    return (array[..., channels], list(channels))

def sliding_windows(array, p_in):
    total = array.shape[0] - p_in
    if total <= 0:
        raise ValueError(f'Not enough timesteps for p_in={p_in}: shape={array.shape}')
    x = np.stack([array[i:i + p_in] for i in range(total)])
    y = np.stack([array[i + p_in] for i in range(total)])
    return (x, y)

def norm_fit(x):
    return (x.min(axis=(0, 1, 2), keepdims=True), x.max(axis=(0, 1, 2), keepdims=True))

def norm_apply(x, mn, mx, eps=1e-06):
    scale = mx - mn
    scale = torch.where(scale < eps, torch.ones_like(scale), scale)
    return (x - mn) / (scale + eps)

def denorm(x, mn, mx, eps=1e-06):
    return mn + x * (mx - mn + eps)

def make_grid_coords_np(h, w):
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32), np.linspace(-1, 1, w, dtype=np.float32), indexing='ij')
    return np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32)

def make_line_coords_np(n):
    x = np.linspace(-1, 1, n, dtype=np.float32)
    return np.stack([np.zeros_like(x), x], axis=-1).astype(np.float32)

class WinDataset(Dataset):

    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index]))

def compute_metrics(y_true, y_pred, eps=1e-08):
    b, _, _, c = y_true.shape
    yt = y_true.reshape(b, -1, c)
    yp = y_pred.reshape(b, -1, c)
    diff = yp - yt
    rmse_g = torch.sqrt((diff ** 2).mean())
    mae_g = diff.abs().mean()
    smape_g = (diff.abs() / (yt.abs() + yp.abs() + eps)).mean()
    r2_g = 1 - (diff ** 2).sum() / ((yt - yt.mean()) ** 2).sum().clamp(min=eps)
    rmse_c = torch.sqrt((diff ** 2).mean(1).mean(0) + eps)
    mae_c = diff.abs().mean(1).mean(0)
    smape_c = (diff.abs() / (yt.abs() + yp.abs() + eps)).mean(1).mean(0)
    ss_tot = ((yt - yt.mean(1, keepdim=True)) ** 2).sum(1).mean(0)
    r2_c = 1 - (diff ** 2).sum(1).mean(0) / (ss_tot + eps)
    return {'per_channel': {'rmse': rmse_c.detach().cpu().tolist(), 'mae': mae_c.detach().cpu().tolist(), 'smape': smape_c.detach().cpu().tolist(), 'r2': r2_c.detach().cpu().tolist()}, 'global': {'rmse': float(rmse_g), 'mae': float(mae_g), 'smape': float(smape_g), 'r2': float(r2_g)}}

def freq_weighted_loss(y, mu):
    yf = torch.fft.rfft2(y.permute(0, 3, 1, 2))
    mf = torch.fft.rfft2(mu.permute(0, 3, 1, 2))
    diff = yf - mf
    h, w = (y.shape[1], y.shape[2])
    hf, wf = (diff.shape[2], diff.shape[3])
    yy = torch.linspace(-1, 1, h, device=y.device).unsqueeze(1).expand(h, w)
    xx = torch.linspace(-1, 1, w, device=y.device).unsqueeze(0).expand(h, w)
    radius = torch.sqrt(yy ** 2 + xx ** 2)[:hf, :wf]
    return (torch.exp(-3.0 * radius).unsqueeze(0).unsqueeze(0) * diff.abs()).mean()

def pinball_loss(y, yh, tau):
    return torch.maximum(tau * (y - yh), (tau - 1) * (y - yh)).mean()

def compute_loss(pred, target, loss_delta, q_w):
    huber = nn.HuberLoss(delta=loss_delta)(pred, target)
    freq = freq_weighted_loss(target, pred)
    quantile = (pinball_loss(target, pred, 0.2) + pinball_loss(target, pred, 0.5) + pinball_loss(target, pred, 0.8)) / 3.0
    return huber + 0.1 * freq + q_w * quantile

def train_one_epoch(model, loader, mn, mx, optimizer, device, loss_delta, q_w):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(norm_apply(x, mn, mx))
        loss = compute_loss(pred, norm_apply(y, mn, mx), loss_delta, q_w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
    return total / len(loader)

@torch.no_grad()
def eval_model(model, loader, mn, mx, device, loss_delta, q_w):
    model.eval()
    total = 0.0
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(norm_apply(x, mn, mx))
        total += float(compute_loss(pred, norm_apply(y, mn, mx), loss_delta, q_w).detach())
        preds.append(denorm(pred, mn, mx).cpu())
        targets.append(y.cpu())
    return (total / len(loader), compute_metrics(torch.cat(targets), torch.cat(preds)))

def prepare_data(train_path, test_path, cfg):
    train_raw = np.load(train_path)
    test_raw = np.load(test_path)
    raw_is_grid = is_grid4d(train_raw)
    train_full = standardize_to_grid_or_node4d(train_raw)
    test_full = standardize_to_grid_or_node4d(test_raw)
    train_full, channels = select_channels(train_full, cfg.channels)
    test_full, _ = select_channels(test_full, channels)
    split = int(train_full.shape[0] * 0.9)
    train_data = train_full[:split]
    val_data = train_full[split:]

    def maybe_crop(array):
        if not cfg.use_crop or not is_grid4d(array):
            return array
        return crop_data(array, cfg.crop_mode, cfg.crop_size, seed=cfg.seed, crop_box=cfg.crop_box)
    train_data = maybe_crop(train_data)
    val_data = maybe_crop(val_data)
    test_data = maybe_crop(test_full)
    coords_tensor = None
    original_grid_shape = None
    if cfg.node_mode:
        if train_data.shape[1] != 1:
            h_grid, w_grid = (int(train_data.shape[1]), int(train_data.shape[2]))
            original_grid_shape = (h_grid, w_grid)

            def grid_to_node(array):
                return array.reshape(array.shape[0], 1, h_grid * w_grid, array.shape[-1])
            train_data = grid_to_node(train_data)
            val_data = grid_to_node(val_data)
            test_data = grid_to_node(test_data)
        n = int(train_data.shape[2])
        if cfg.coords_npy:
            coords_np = np.load(cfg.coords_npy).astype(np.float32)
            if coords_np.shape[1] != 2:
                coords_np = coords_np[:, :2]
        elif raw_is_grid and original_grid_shape is not None:
            coords_np = make_grid_coords_np(*original_grid_shape)
        else:
            coords_np = make_line_coords_np(n)
        coords_tensor = torch.from_numpy(coords_np)
        h_model, w_model = (1, n)
    else:
        h_model = int(train_data.shape[1])
        w_model = int(train_data.shape[2])
        if cfg.coords_npy:
            coords_np = np.load(cfg.coords_npy).astype(np.float32)
            if coords_np.shape[1] != 2:
                coords_np = coords_np[:, :2]
        else:
            coords_np = make_grid_coords_np(h_model, w_model)
        coords_tensor = torch.from_numpy(coords_np)
    x_train, y_train = sliding_windows(train_data, cfg.p_in)
    x_val, y_val = sliding_windows(val_data, cfg.p_in)
    x_test, y_test = sliding_windows(test_data, cfg.p_in)
    min_np, max_np = norm_fit(train_data)
    mn = torch.from_numpy(min_np.astype(np.float32)).to(cfg.device)
    mx = torch.from_numpy(max_np.astype(np.float32)).to(cfg.device)
    return {'train': (x_train, y_train), 'val': (x_val, y_val), 'test': (x_test, y_test), 'mn': mn, 'mx': mx, 'coords': coords_tensor, 'h': h_model, 'w': w_model, 'c': len(channels), 'channels': channels}

def run_training(train_path, test_path, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg.seed)
    data = prepare_data(train_path, test_path, cfg)
    loader_kwargs = {'num_workers': cfg.num_workers, 'pin_memory': cfg.pin_memory}
    x_train, y_train = data['train']
    x_val, y_val = data['val']
    x_test, y_test = data['test']
    train_loader = DataLoader(WinDataset(x_train, y_train), batch_size=cfg.batch, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(WinDataset(x_val, y_val), batch_size=cfg.batch, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(WinDataset(x_test, y_test), batch_size=cfg.batch, shuffle=False, **loader_kwargs)
    model = BPRNet(h=data['h'], w=data['w'], c=data['c'], p=cfg.p_in, d_model=cfg.d_model, periods=cfg.periods, num_frequencies=cfg.num_frequencies, dropout=cfg.dropout, num_prototypes=cfg.num_prototypes, routing_epsilon=cfg.routing_epsilon, routing_temperature=cfg.routing_temperature, use_latent_distance=cfg.use_latent_distance, bpa_init_alpha=cfg.bpa_init_alpha, node_mode=cfg.node_mode, coords=data['coords'], use_periodic=cfg.use_periodic, use_context=cfg.use_context, residual_to_last=cfg.residual_to_last).to(cfg.device)
    fast_params = [p for name, p in model.named_parameters() if 'log_alpha' in name]
    other_params = [p for name, p in model.named_parameters() if 'log_alpha' not in name]
    optimizer = torch.optim.AdamW([{'params': other_params, 'lr': cfg.lr}, {'params': fast_params, 'lr': cfg.lr * 10}], weight_decay=cfg.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_path = os.path.join(save_dir, 'bprnet_best.pt')
    best_rmse = float('inf')
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, data['mn'], data['mx'], optimizer, cfg.device, cfg.loss_delta, cfg.q_w)
        val_loss, val_metrics = eval_model(model, val_loader, data['mn'], data['mx'], cfg.device, cfg.loss_delta, cfg.q_w)
        scheduler.step()
        rmse = val_metrics['global']['rmse']
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({'model': model.state_dict(), 'config': asdict(cfg), 'min': data['mn'].cpu().numpy(), 'max': data['mx'].cpu().numpy(), 'channels': data['channels'], 'h': data['h'], 'w': data['w'], 'c': data['c'], 'coords': data['coords'].numpy() if data['coords'] is not None else None}, best_path)
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | RMSE={rmse:.6f} | alpha={model.fapr.get_alpha():.4f} | {elapsed:.1f}s')
    checkpoint = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    predictions = []
    targets = []
    inference_start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device, non_blocking=True)
            pred = model(norm_apply(x, data['mn'], data['mx']))
            predictions.append(denorm(pred, data['mn'], data['mx']).cpu())
            targets.append(y)
    inference_time = time.time() - inference_start
    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)
    metrics = compute_metrics(y_true, y_pred)
    np.save(os.path.join(save_dir, 'y_pred_test.npy'), y_pred.numpy())
    np.save(os.path.join(save_dir, 'y_true_test.npy'), y_true.numpy())
    result = {'best_val_rmse': best_rmse, 'test_metrics': metrics, 'parameters': sum((p.numel() for p in model.parameters())), 'train_seconds': time.time() - start, 'inference_seconds': inference_time, 'alpha': model.fapr.get_alpha()}
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return (model, result)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-npy', required=True)
    parser.add_argument('--test-npy', required=True)
    parser.add_argument('--save-dir', default='./outputs')
    parser.add_argument('--p-in', type=int, default=25)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-frequencies', type=int, default=13)
    parser.add_argument('--num-prototypes', type=int, default=128)
    parser.add_argument('--routing-epsilon', type=float, default=0.08)
    parser.add_argument('--routing-temperature', type=float, default=1.4)
    parser.add_argument('--bpa-init-alpha', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--loss-delta', type=float, default=0.8)
    parser.add_argument('--q-w', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--pin-memory', type=str2bool, default=True)
    parser.add_argument('--node-mode', type=str2bool, default=True)
    parser.add_argument('--use-crop', type=str2bool, default=False)
    parser.add_argument('--crop-mode', choices=['none', 'center', 'top_left', 'bottom_right', 'random'], default='center')
    parser.add_argument('--crop-size', type=int, default=20)
    parser.add_argument('--crop-box', type=int, nargs=4, default=None)
    parser.add_argument('--channels', type=int, nargs='+', default=None)
    parser.add_argument('--coords-npy', default=None)
    parser.add_argument('--use-periodic', type=str2bool, default=True)
    parser.add_argument('--use-context', type=str2bool, default=True)
    parser.add_argument('--residual-to-last', type=str2bool, default=True)
    parser.add_argument('--device', default=None)
    return parser

def main():
    args = build_parser().parse_args()
    cfg = TrainConfig(p_in=args.p_in, batch=args.batch, epochs=args.epochs, lr=args.lr, wd=args.wd, num_workers=args.num_workers, pin_memory=args.pin_memory, d_model=args.d_model, num_frequencies=args.num_frequencies, dropout=args.dropout, loss_delta=args.loss_delta, seed=args.seed, use_periodic=args.use_periodic, use_context=args.use_context, residual_to_last=args.residual_to_last, num_prototypes=args.num_prototypes, routing_epsilon=args.routing_epsilon, routing_temperature=args.routing_temperature, bpa_init_alpha=args.bpa_init_alpha, q_w=args.q_w, node_mode=args.node_mode, use_crop=args.use_crop, crop_mode=args.crop_mode, crop_size=args.crop_size, crop_box=tuple(args.crop_box) if args.crop_box is not None else None, channels=args.channels, coords_npy=args.coords_npy, device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run_training(args.train_npy, args.test_npy, args.save_dir, cfg)
if __name__ == '__main__':
    main()
