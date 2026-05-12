import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.model import NPPModel
from core.config import get_memristor_config
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

def run_benchmark(num_samples=1000, tolerance=0.05, max_steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NPPModel().to(device)
    model.load_state_dict(torch.load('./npp_final.pth', map_location=device))
    model.eval()

    config = get_memristor_config()
    layer = AnalogLinear(1, 1, bias=False, rpu_config=config)

    success_count = 0
    total_steps = 0
    final_errors = []

    print(f"Starting benchmark: {num_samples} devices.")
    print(f"Max steps: {max_steps} | Tolerance: ±{tolerance}")

    for i in range(num_samples):

        g_current = np.random.uniform(-1.0, 1.0) 
        g_target = np.random.uniform(-1.0, 1.0)
        
        layer.set_weights(torch.tensor([[g_current]]))
        layer.train()
        
        step = 0
        while step < max_steps:
            step += 1
            delta_g_req = g_target - g_current
            
            # Проверка успешного достижения цели
            if abs(delta_g_req) <= tolerance:
                success_count += 1
                break
            
            gs_tensor = torch.tensor([[g_current]], dtype=torch.float32).to(device)
            dg_tensor = torch.tensor([[delta_g_req]], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                t_pulse = model(gs_tensor, dg_tensor).item()
            
            t_pulse = np.clip(t_pulse, 0.01, 1.0)
            
            optimizer = AnalogSGD(layer.parameters(), lr=t_pulse)
            direction = 1.0 if delta_g_req > 0 else -1.0
            
            x = torch.tensor([[1.0]])
            y_pred = layer(x)
            loss = y_pred * (-direction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            g_current = layer.get_weights()[0].item()
        
        total_steps += step
        final_errors.append(abs(g_target - g_current))
        
        if (i + 1) % 200 == 0:
            print(f"Tested {i + 1}/{num_samples}")

    print("\n================================================")
    print("================================================")
    print(f"Success Rate:        {success_count/num_samples*100:.2f}%")
    print(f"Average Steps:       {total_steps/num_samples:.2f}")
    print(f"Mean Final Error:    {np.mean(final_errors):.4f}")
    print("================================================")

if __name__ == "__main__":
    run_benchmark()