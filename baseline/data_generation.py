import torch
import os
import sys

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.vcm_simulator import VCMSimulator

def generate_dataset(num_samples=10000, batch_size=1000, save_dir='data'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = VCMSimulator(device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    dataset = []
    
    print(f"Generating {num_samples} samples. Device: {device}")
    
    samples_generated = 0
    while samples_generated < num_samples:
        current_batch = min(batch_size, num_samples - samples_generated)
        
        g_start = torch.empty(current_batch, 1, device=device).uniform_(simulator.g_min, simulator.g_max)
        g_target = torch.empty(current_batch, 1, device=device).uniform_(simulator.g_min, simulator.g_max)
        
        delta_g = g_target - g_start
        valid_mask = torch.abs(delta_g) > 0.05 * (simulator.g_max - simulator.g_min)
        
        g_start = g_start[valid_mask].view(-1, 1)
        g_target = g_target[valid_mask].view(-1, 1)
        delta_g = delta_g[valid_mask].view(-1, 1)
        
        if g_start.shape[0] == 0:
            continue
            
        direction = torch.sign(delta_g)
        i_t = torch.where(direction > 0, torch.ones_like(direction), -torch.ones_like(direction))
        

        t_pulse = simulator.simulate_until_target(g_start, g_target, i_t, max_time=1e-6)
        
        reached_mask = (t_pulse < 1e-6)
        g_start = g_start[reached_mask].view(-1, 1)
        g_target = g_target[reached_mask].view(-1, 1)
        i_t = i_t[reached_mask].view(-1, 1)
        t_pulse = t_pulse[reached_mask].view(-1, 1)
        direction = direction[reached_mask].view(-1, 1)
        
        if g_start.shape[0] == 0:
            continue
            
        t_history_max = 2 * t_pulse
        _, g_history_tensor = simulator.simulate_pulse(g_start, i_t, t_history_max, record_history=True)
        
        for i in range(g_start.shape[0]):
            steps = int((t_history_max[i].item() / simulator.dt)) + 1
            sample_history = g_history_tensor[i, :steps].cpu()
            
            dataset.append({
                'g_start': g_start[i].item(),
                'g_target': g_target[i].item(),
                'direction': direction[i].item(),
                't_pulse': t_pulse[i].item(),
                'g_history': sample_history
            })
            
        samples_generated = len(dataset)
        print(f"Собрано: {samples_generated}/{num_samples}")

    dataset = dataset[:num_samples]

    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    torch.save(train_data, os.path.join(save_dir, 'train.pt'))
    torch.save(val_data, os.path.join(save_dir, 'val.pt'))
    torch.save(test_data, os.path.join(save_dir, 'test.pt'))
    
    print("Saved to", save_dir)
    print(f"Size: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

if __name__ == "__main__":
    with torch.no_grad():
        generate_dataset(num_samples=1000000, batch_size=1000)