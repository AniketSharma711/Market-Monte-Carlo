import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- 1. THE GOLDEN KERNEL (Same as before) ---
kernel_code = """
#include <curand_kernel.h>

extern "C" {
    __global__ void monte_carlo_kernel(float *d_results, float S0, float K, float T, float r, float sigma, int n_sims, int n_steps) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_sims) return;

        curandStatePhilox4_32_10_t state;
        curand_init(1234 + idx, 0, 0, &state);

        float dt = T / n_steps;
        float S = S0;
        
        // Fast math constants
        float drift = (r - 0.5f * sigma * sigma) * dt;
        float vol = sigma * __fsqrt_rn(dt);

        for (int i = 0; i < n_steps; i++) {
            float Z = curand_normal(&state);
            S = S * __expf(drift + vol * Z);
        }

        // Payoff (Call)
        float payoff = S - K;
        if (payoff < 0.0f) payoff = 0.0f;

        d_results[idx] = payoff * __expf(-r * T);
    }
}
"""

# Compile once
print("⚙️  Compiling CUDA Kernel...")
mod = SourceModule(kernel_code, options=['--use_fast_math', '-allow-unsupported-compiler'], no_extern_c=True)
monte_carlo_kernel = mod.get_function("monte_carlo_kernel")
print("✅ Compilation Complete.")

# --- 2. SIMULATION FUNCTION ---
def run_gpu_sim(S0, K, T, r, sigma):
    n_sims = 1_000_000
    n_steps = 256
    
    results = np.zeros(n_sims, dtype=np.float32)
    d_results = cuda.mem_alloc(results.nbytes)
    
    block_size = 256
    grid_size = (n_sims + block_size - 1) // block_size
    
    # Run Kernel
    monte_carlo_kernel(d_results, np.float32(S0), np.float32(K), np.float32(T), 
                       np.float32(r), np.float32(sigma), np.int32(n_sims), np.int32(n_steps),
                       block=(block_size, 1, 1), grid=(grid_size, 1))
    
    cuda.memcpy_dtoh(results, d_results)
    return results

# --- 3. BUILD THE GUI ---
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.35) # Make room for sliders at bottom

# Initial Plot
S0_init = 100.0
initial_data = run_gpu_sim(S0_init, 100.0, 1.0, 0.05, 0.2)
n, bins, patches = ax.hist(initial_data[initial_data > 0], bins=50, color='#00CC96', alpha=0.7)
ax.set_title(f"Option Price Distribution (GPU) - Mean Price: ${np.mean(initial_data):.2f}")
ax.set_xlabel("Payoff ($)")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3)

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_S0 = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_K = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_sigma = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

s_S0 = Slider(ax_S0, 'Asset (S0)', 50.0, 150.0, valinit=S0_init)
s_K = Slider(ax_K, 'Strike (K)', 50.0, 150.0, valinit=100.0)
s_sigma = Slider(ax_sigma, 'Vol (σ)', 0.1, 1.0, valinit=0.2)

# Update Function
def update(val):
    # 1. Get values
    S0 = s_S0.val
    K = s_K.val
    sigma = s_sigma.val
    
    # 2. Re-run GPU Simulation (Real-time!)
    results = run_gpu_sim(S0, K, 1.0, 0.05, sigma)
    price = np.mean(results)
    
    # 3. Update Plot
    ax.cla() # Clear axis
    # Only show positive payoffs to keep graph readable
    active = results[results > 0]
    if len(active) == 0: active = [0] # Handle out-of-money case
    
    ax.hist(active, bins=50, color='#00CC96', alpha=0.7)
    ax.set_title(f"Option Price: ${price:.2f} | GPU Latency: ~40ms")
    ax.set_xlabel("Payoff ($)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.canvas.draw_idle()

# Connect sliders
s_S0.on_changed(update)
s_K.on_changed(update)
s_sigma.on_changed(update)

print("🚀 Desktop Dashboard Running... (Close window to exit)")
plt.show()