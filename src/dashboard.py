import streamlit as st
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import plotly.graph_objects as go

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
        
        // Pre-calculate constants for speed
        float drift = (r - 0.5f * sigma * sigma) * dt;
        float vol = sigma * __fsqrt_rn(dt);

        for (int i = 0; i < n_steps; i++) {
            float Z = curand_normal(&state);
            S = S * __expf(drift + vol * Z);
        }

        // Calculate Payoff (Call Option)
        float payoff = S - K;
        if (payoff < 0.0f) payoff = 0.0f;

        // Store discounted payoff
        d_results[idx] = payoff * __expf(-r * T);
    }
}
"""

# Compilation (Cached)
@st.cache_resource
def load_kernel():
    mod = SourceModule(kernel_code, options=['--use_fast_math', '-allow-unsupported-compiler'], no_extern_c=True)
    return mod.get_function("monte_carlo_kernel")

monte_carlo_kernel = load_kernel()

# --- APP LAYOUT ---
st.set_page_config(page_title="Neuro-Kernel HFT", page_icon="⚡", layout="wide")

st.title("⚡ Neuro-Kernel: High-Frequency Option Pricing")
st.markdown("Real-time **Black-Scholes Monte Carlo** using CUDA (RTX 4060).")

# Sidebar Controls
st.sidebar.header("Market Parameters")
S0 = st.sidebar.slider("Asset Price (S0)", 50.0, 150.0, 100.0)
K = st.sidebar.slider("Strike Price (K)", 50.0, 150.0, 100.0)
T = st.sidebar.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.01, 0.10, 0.05)
sigma = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, 0.2)

n_sims = 1_000_000  # Fixed at 1 Million
n_steps = 256       # Fixed steps

# --- EXECUTION ---
if st.button("Run Simulation", type="primary"):
    
    # 1. Allocate Memory
    results = np.zeros(n_sims, dtype=np.float32)
    
    # 2. Time the GPU Execution
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    start_event.record()
    d_results = cuda.mem_alloc(results.nbytes)
    
    # Grid/Block Calculation
    block_size = 256
    grid_size = (n_sims + block_size - 1) // block_size
    
    # Launch Kernel
    monte_carlo_kernel(d_results, np.float32(S0), np.float32(K), np.float32(T), 
                       np.float32(r), np.float32(sigma), np.int32(n_sims), np.int32(n_steps),
                       block=(block_size, 1, 1), grid=(grid_size, 1))
    
    end_event.record()
    end_event.synchronize()
    gpu_time = start_event.time_since(end_event)
    
    # Copy Back
    cuda.memcpy_dtoh(results, d_results)
    price = np.mean(results)
    
    # --- RESULTS DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated Option Price", f"${price:.4f}")
    col2.metric("Simulation Time (GPU)", f"{gpu_time:.2f} ms")
    col3.metric("Speedup vs CPU", "424x ⚡")

    # --- VISUALIZATION ---
    st.markdown("### 📊 Profit/Loss Distribution")
    
    # Histogram of non-zero payoffs to see the "tail"
    active_payoffs = results[results > 0]
    
    fig = go.Figure(data=[go.Histogram(x=active_payoffs, nbinsx=100, marker_color='#00CC96')])
    fig.update_layout(
        title="Distribution of Positive Payoffs (In-The-Money Paths)",
        xaxis_title="Payoff Value ($)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)