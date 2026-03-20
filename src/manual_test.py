from judge_gpu import judge_ai_kernel

# This is the "Perfect" code that the AI is struggling to write.
# It uses 'extern "C"' to prevent name-mangling errors.
golden_code = """
#include <curand_kernel.h>

extern "C" {
    __global__ void monte_carlo_kernel(float *d_results, float S0, float K, float T, float r, float sigma, int n_sims, int n_steps) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < n_sims) {
            // 1. Initialize Random Number Generator (Expensive but necessary)
            curandStatePhilox4_32_10_t state;
            curand_init(1234, idx, 0, &state);
            
            // 2. Simulation Variables
            float dt = T / n_steps;
            float current_price = S0;
            
            // 3. The Path Loop (365 days)
            for(int i = 0; i < n_steps; i++) {
                float z = curand_normal(&state);
                current_price *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
            }
            
            // 4. Store Result
            d_results[idx] = current_price;
        }
    }
}
"""

print("🚀 Running Manual Test with Golden Kernel...")
success, msg, price = judge_ai_kernel(golden_code)

if success:
    print(f"SUCCESS! The GPU is working perfectly.")
    print(f"Time: {msg}") # This should be ~20-50ms
    print(f"Price: ${price:.2f} (Should be around $8.00)")
else:
    print(f"FAIL: {msg}")