import numpy as np
import time

def cpu_monte_carlo(S0, K, T, r, sigma, n_sims, n_steps):
    dt = T / n_steps
    Z = np.random.normal(0, 1, (n_sims, n_steps))
    
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    # Calculate price step-by-step
    for t in range(1, n_steps + 1):
        # The Math Formula in Python
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
    # Calculate Option Price (Payoff at the end)
    payoffs = np.maximum(paths[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

if __name__ == "__main__":
    S0, K = 100, 105    # Stock Price $100, Strike Price $105
    T, r, sigma = 1.0, 0.05, 0.2 # 1 Year, 5% Risk Free, 20% Volatility
    N_SIMS = 1_000_000  # 1 Million Paths
    N_STEPS = 365       # 365 Days
    
    print(f"Running {N_SIMS} simulations on CPU...")
    start = time.time()
    price = cpu_monte_carlo(S0, K, T, r, sigma, N_SIMS, N_STEPS)
    end = time.time()
    
    print(f"Fair Price: ${price:.2f}")
    print(f"CPU Time: {end - start:.4f} seconds")