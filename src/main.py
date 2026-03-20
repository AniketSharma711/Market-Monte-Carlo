import os
import time
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types

# CONFIGURATION
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Please check the .env file.")

client = genai.Client(api_key=API_KEY)

MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash"]

def robust_generate(formatted_contents, system_instruction=None):
    """
    Tries multiple valid models. If one hits a rate limit, swaps to the next.
    """
    for attempt in range(3): # Try the list 3 times
        for model in MODELS:
            try:
                print(f" Thinking with {model}...")
                
                response = client.models.generate_content(
                    model=model,
                    contents=formatted_contents, # Passing strictly typed objects
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        system_instruction=system_instruction
                    )
                )
                return response # Success!
                
            except Exception as e:
                error_str = str(e)
                # Check for rate limits (429) or quota issues
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f"{model} busy/exhausted. Switching...")
                    continue # Try next model immediately
                else:
                    print(f" Error with {model}: {error_str}")
                    # If it's a real error (not quota), we still try the next model just in case
                    continue

        print("All models busy. Sleeping for 20 seconds...")
        time.sleep(20)
    
    raise RuntimeError("Failed to generate response after multiple retries.")

#  SYSTEM INSTRUCTIONS 
SYSTEM_PROMPT = """
You are an expert CUDA C++ developer. 
Task: Write a Python script using PyCUDA to implement a Black-Scholes Monte Carlo kernel.
- Use __expf, __sqrtf, and curand_kernel.h.
- Include the #include <curand_kernel.h> OUTSIDE the extern "C" block.
- RETURN ONLY THE RAW PYTHON CODE. NO MARKDOWN. NO EXPLANATIONS.
"""

#  MAIN AGENT LOOP 
def main():
    print("AI Optimization Loop Starting...")
    
    # Local history (simple dicts for easy management)
    chat_history = [] 
    
    # The starting prompt
    current_prompt = "Write a complete Python script with PyCUDA that implements a Monte Carlo Black-Scholes kernel. Include strict error checking."

    for iteration in range(1, 4): # Run 3 optimization cycles
        print(f"\n🔄 Iteration {iteration}/3")
        
        api_contents = []
        
        # Convert previous history
        for msg in chat_history:
            api_contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
            
        # Add the current user prompt
        api_contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=current_prompt)]
            )
        )

        try:
            # Generate code using the strictly typed list
            response = robust_generate(api_contents, SYSTEM_PROMPT)
            
            # Clean up Markdown wrappers
            code = response.text.replace("```python", "").replace("```", "").strip()
            
        except Exception as e:
            print(f"Critical AI Failure: {e}")
            break

        # Save the generated code
        filename = "src/generated_kernel.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Code saved to {filename}")

        # Compile & Test
        print("Compiling & Benchmarking...")
        result = subprocess.run(
            ["python", filename], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("Success! Kernel is working.")
            # Save the winner
            with open("src/golden_kernel.py", "w") as f:
                f.write(code)
            break # Exit loop
        else:
            print("Compilation/Runtime Error.")
            error_msg = result.stderr + result.stdout
            print(f"DEBUG ERROR: {error_msg[:300]}...") # Print first 300 chars of error
            
            # Feed error back to AI for the next turn
            chat_history.append({"role": "user", "content": current_prompt})
            chat_history.append({"role": "model", "content": code})
            
            current_prompt = f"The previous code failed with this error:\n{error_msg}\n\nFix the code and return the full corrected script."

if __name__ == "__main__":
    main()