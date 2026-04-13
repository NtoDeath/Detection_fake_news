#!/usr/bin/env python3
"""
Setup script for Fake News Detection CLI Tool
Works on: Windows, macOS, Linux
Usage: python setup.py
"""

import subprocess
import sys
import os
import shutil


def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    try:
        if description:
            print(f"  {description}...", end=" ", flush=True)
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        if description:
            print("✅")
        return result.stdout
    except subprocess.CalledProcessError as e:
        if description:
            print("❌")
        print(f"Error: {e.stderr}")
        return None


def main():
    print("\n🔍 Fake News Detection CLI - Setup")
    print("====================================\n")
    
    # 1. Check Python
    print("1️⃣  Checking Python...")
    python_version = run_command(f"{sys.executable} --version")
    if python_version:
        print(f"   Found: {python_version.strip()}")
    else:
        print("   ❌ Python not found!")
        sys.exit(1)
    print("")
    
    # 2. Install dependencies
    print("2️⃣  Installing dependencies...")
    print("   (This may take 1-2 minutes)")
    cmd = f"{sys.executable} -m pip install -r requirements.txt --quiet"
    if run_command(cmd):
        print("   ✅ Dependencies installed\n")
    else:
        print("   ❌ Failed to install dependencies")
        sys.exit(1)
    
    # 3. Check/Copy models
    print("3️⃣  Checking models...")
    if os.path.isdir("models"):
        print("   ✅ Models directory found\n")
    else:
        print("   ⚠️  Models directory not found")
        if os.path.isfile("models_copy.py"):
            print("   Attempting to copy models...")
            if run_command(f"{sys.executable} models_copy.py"):
                print("   ✅ Models copied\n")
            else:
                print("   ⚠️  Could not copy models (they may be missing)\n")
        else:
            print("   Creating empty models directory...")
            os.makedirs("models", exist_ok=True)
            print("   ✅ Created\n")
    
    # 4. Verify setup
    print("4️⃣  Verifying setup...")
    verify_code = """
try:
    from main import app
    print('   ✅ CLI verified')
except Exception as e:
    print(f'   ❌ Error: {e}')
    exit(1)
"""
    if run_command(f'{sys.executable} -c "{verify_code}"'):
        print("")
    else:
        print("   ❌ Verification failed")
        sys.exit(1)
    
    # Success message
    print("✅ Setup complete!\n")
    print("Ready to use! Run:")
    print("  python main.py              (interactive REPL mode)")
    print("  python main.py predict \"Your text here\"")
    print("  python main.py info\n")


if __name__ == "__main__":
    main()
