import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_memristor_config, G_MIN, G_MAX
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

def generate_raw_dataset(num_samples: int = 1000000, save_path: str = 'baseline/raw_data.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config = get_memristor_config()
    
    layer = AnalogLinear(1, 1, bias=False, rpu_config=config)
    
    g_start_list = []
    delta_g_list = []
    t_pulse_list = []

    layer.train()
    print(f"Star generation {num_samples} samples with SoftBoundsDevice...")

    for i in range(num_samples):
        g_start = torch.empty(1, 1).uniform_(G_MIN, G_MAX)
        layer.set_weights(g_start)
        
        t_pulse = np.random.uniform(0.01, 1.0)
        optimizer = AnalogSGD(layer.parameters(), lr=t_pulse)
        
        direction = 1.0 if np.random.rand() > 0.5 else -1.0
        
        x = torch.tensor([[1.0]])
        y_pred = layer(x)
        
        loss = y_pred * (-direction) 
        # ---------------------------------------------------------
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        g_final = layer.get_weights()[0].item()
        delta_g = g_final - g_start.item()
        
        g_start_list.append(g_start.item())
        delta_g_list.append(delta_g)
        t_pulse_list.append(t_pulse)

    dataset = {
        'g_start': torch.tensor(g_start_list, dtype=torch.float32),
        'delta_g': torch.tensor(delta_g_list, dtype=torch.float32),
        't_pulse': torch.tensor(t_pulse_list, dtype=torch.float32)
    }

    torch.save(dataset, save_path)
    print(f"Data saved: {save_path}")

if __name__ == "__main__":
    generate_raw_dataset()