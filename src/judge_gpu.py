import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import time

def judge_ai_kernel(kernel_code, debug_mode=False):
    """
    Compiles and runs the AI's CUDA kernel.
    Returns: (success_bool, message_or_time, average_price)
    """
    
    # --- 1. CONFIGURATION ---
    # Must match your CPU baseline settings for fair comparison
    N_SIMS = 1_000_000
    N_STEPS = 365
    S0 = np.float32(100.0)
    K = np.float32(105.0)
    T = np.float32(1.0)
    R = np.float32(0.05)
    SIGMA = np.float32(0.2)

    # --- 2. COMPILATION ---
    try:
        # We tell PyCUDA to compile the code. 
        # no_extern_c=True means we expect standard C++ mangling (or extern "C" in code)
        mod = SourceModule(kernel_code, options=['--use_fast_math', '-allow-unsupported-compiler'], no_extern_c=True)
        
        # We assume the AI will ALWAYS name the function 'monte_carlo_kernel'
        func = mod.get_function("monte_carlo_kernel")
    except Exception as e:
        # If compilation fails, return the compiler error so the AI can fix itpt
        return False, f"Compilation Failed:\n{str(e)}", 0.0

    # --- 3. MEMORY ALLOCATION ---
    try:
        # Allocate array on GPU for the final prices (1M floats = ~4MB)
        # We don't need to upload input data arrays, just scalars.
        d_results = drv.mem_alloc(N_SIMS * 4) # 4 bytes per float32
    except Exception as e:
        return False, f"Memory Allocation Failed: {str(e)}", 0.0

    # --- 4. EXECUTION ---
    start_event = drv.Event()
    end_event = drv.Event()

    try:
        # Calculate Grid/Block sizes
        # 256 threads per block is a standard "safe" number for GPUs
        block_dim = (256, 1, 1)
        grid_dim = (int((N_SIMS + 256 - 1) // 256), 1)

        start_event.record()
        
        # CALL THE KERNEL
        # Note: The arguments must match the C++ signature exactly!
        # void monte_carlo_kernel(float *results, float S0, float K, float T, float r, float sigma, int n_sims, int n_steps)
        func(d_results, S0, K, T, R, SIGMA, np.int32(N_SIMS), np.int32(N_STEPS),
             block=block_dim, grid=grid_dim)
             
        end_event.record()
        end_event.synchronize()
        
        # Time in milliseconds
        gpu_time_ms = start_event.time_till(end_event)
        
    except Exception as e:
        return False, f"Runtime Error (Kernel Crash): {str(e)}", 0.0

    # --- 5. VERIFICATION ---
    # Copy results back to CPU to check if the math is right
    h_results = np.empty(N_SIMS, dtype=np.float32)
    drv.memcpy_dtoh(h_results, d_results)

    # Calculate option price (Logic: Max(Price - Strike, 0))
    payoffs = np.maximum(h_results - K, 0)
    price = np.exp(-R * T) * np.mean(payoffs)

    # Sanity Check: If price is NaN, negative, or wildly huge (> $1000), the math failed.
    if np.isnan(price) or price < 0 or price > 1000:
         return False, f"Math Failed: Calculated Price was ${price:.2f} (Expected ~$8.00)", price

    return True, f"{gpu_time_ms:.4f} ms", price

# --- TEST BLOCK ---
# You can run this file directly to test if PyCUDA is working.
if __name__ == "__main__":
    print("Checking PyCUDA setup...")
    # A dummy kernel that just returns '100.0' for every price
    # This proves your GPU is listening.
    test_code = """
    __global__ void monte_carlo_kernel(float *results, float S0, float K, float T, float r, float sigma, int n_sims, int n_steps) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n_sims) {
            results[idx] = 113.0f; // Fake result to test memory copy
        }
    }
    """
    success, msg, price = judge_ai_kernel(test_code, debug_mode=True)
    if success:
        print(f"✅ Setup Working! Test Kernel Time: {msg}")
        print(f"   (The price will be wrong because we hardcoded 113.0: Price=${price:.2f})")
    else:
        print(f"❌ Setup Failed: {msg}")