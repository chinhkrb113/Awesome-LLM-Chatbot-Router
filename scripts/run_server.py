import os
import sys
import subprocess

if __name__ == "__main__":
    # Set env vars
    env = os.environ.copy()
    env["VECTOR_STORE"] = "memory"
    
    # Add current directory to PYTHONPATH
    cwd = os.getcwd()
    env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Starting Hybrid Router V2 (Default) in {cwd}...")
    
    # Run uvicorn as a module
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nStopped.")
