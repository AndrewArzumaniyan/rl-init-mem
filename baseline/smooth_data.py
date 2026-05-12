import torch
import pandas as pd
import numpy as np
import os

def apply_gradient_smoothing(raw_data_path='baseline/raw_data.pt', output_dir='baseline/'):
    raw_data = torch.load(raw_data_path)
    df = pd.DataFrame({
        'g_start': raw_data['g_start'].numpy(),
        'delta_g': raw_data['delta_g'].numpy(),
        't_pulse': raw_data['t_pulse'].numpy()
    })

    df['g_bin'] = pd.qcut(df['g_start'], q=20, labels=False)
    
    windows = [501, 101, 21, 1]

    for window in windows:
        if window == 1:
            smoothed_df = df.copy()
        else:
            smoothed_chunks = []
            for (bin_id, direction), group in df.groupby(['g_bin', df['delta_g'] >= 0]):
                group = group.sort_values('t_pulse')
                group['delta_g'] = group['delta_g'].rolling(window=window, center=True, min_periods=1).mean()
                smoothed_chunks.append(group)
            
            smoothed_df = pd.concat(smoothed_chunks)

        dataset = {
            'g_start': torch.tensor(smoothed_df['g_start'].values, dtype=torch.float32),
            'delta_g': torch.tensor(smoothed_df['delta_g'].values, dtype=torch.float32),
            't_pulse': torch.tensor(smoothed_df['t_pulse'].values, dtype=torch.float32)
        }
        torch.save(dataset, os.path.join(output_dir, f'smoothed_data_w{window}.pt'))
        print(f"Window {window} is ready")

if __name__ == "__main__":
    apply_gradient_smoothing()