import torch
import numpy as np

def verify_dataset(path='baseline/raw_data.pt'):
    data = torch.load(path)
    g_start = data['g_start'].numpy()
    delta_g = data['delta_g'].numpy()
    t_pulse = data['t_pulse'].numpy()

    print(f"--- Dataset Audit: {path} ---")
    print(f"Samples: {len(g_start)}")
    
    for name, arr in [("G_start", g_start), ("Delta_G", delta_g), ("T_pulse", t_pulse)]:
        print(f"{name:8} | Min: {arr.min():.4f} | Max: {arr.max():.4f} | Mean: {arr.mean():.4f} | Std: {arr.std():.4f}")
    
    nan_count = np.isnan(delta_g).sum()
    zero_delta = (delta_g == 0).sum()
    print(f"NaNs: {nan_count} | Zero Deltas: {zero_delta} ({zero_delta/len(delta_g)*100:.2f}%)")

if __name__ == "__main__":
    verify_dataset()