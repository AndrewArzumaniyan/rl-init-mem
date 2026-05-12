from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

G_MIN = -1.5
G_MAX = 1.5

def get_memristor_config():
    device = LinearStepDevice()

    device.w_min = G_MIN
    device.w_max = G_MAX
    device.dw_min = 0.05

    device.up_down = 0.5       
    device.up_down_dtod = 0.1  

    device.gamma_up = 2.0      
    device.gamma_down = 2.0    
    
    device.dw_min_std = 0.4    
    device.write_noise_std = 0.15 
    device.mult_noise = True 

    device.w_max_dtod = 0.15
    device.w_min_dtod = 0.15
    device.dw_min_dtod = 0.2

    config = SingleRPUConfig(device=device)
    return config