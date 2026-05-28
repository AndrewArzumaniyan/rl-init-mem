import torch

class VCMSimulator:
    def __init__(self, g_min=0.01, g_max=1.0, alpha=0.15, beta=0.005, dt=1e-8, device='cpu'):
        self.g_min = g_min
        self.g_max = g_max
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.device = device

    def step(self, g_t, i_t):
        xi = torch.randn_like(g_t, device=self.device)

        delta_g = self.alpha * i_t * (self.g_max - g_t) * (g_t - self.g_min)
        noise = self.beta * torch.abs(i_t) * xi
        
        g_t_plus_1 = g_t + delta_g + noise
        return torch.clamp(g_t_plus_1, self.g_min, self.g_max)

    def simulate_until_target(self, g_start, g_target, i_t, max_time=2e-6):
        max_steps = int(max_time / self.dt)
        batch_size = g_start.shape[0]
        
        g_curr = g_start.clone()
        direction = torch.sign(g_target - g_start)
        
        crossed = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
        t_pulse = torch.full((batch_size, 1), max_time, device=self.device)
        
        for step_idx in range(1, max_steps + 1):
            g_next = self.step(g_curr, i_t)
            g_curr = torch.where(crossed, g_curr, g_next)
            
            just_crossed = (direction * (g_curr - g_target)) >= 0
            newly_crossed = just_crossed & ~crossed
            
            if newly_crossed.any():
                t_pulse[newly_crossed] = step_idx * self.dt
                crossed = crossed | newly_crossed
                
            if crossed.all():
                break
                
        return t_pulse

    def simulate_pulse(self, g_start, i_t, t_pulse, record_history=False):
        max_steps_tensor = (t_pulse / self.dt).long()
        max_steps = max_steps_tensor.max().item()
        
        g_curr = g_start.clone()
        history = [g_curr.clone()] if record_history else None
        
        for step_idx in range(1, max_steps + 1):
            active_mask = (step_idx <= max_steps_tensor)
            if not active_mask.any():
                break
                
            g_next = self.step(g_curr, i_t)
            g_curr = torch.where(active_mask, g_next, g_curr)
            
            if record_history:
                history.append(g_curr.clone())
                
        if record_history:
            return g_curr, torch.stack(history, dim=1)
        
        return g_curr