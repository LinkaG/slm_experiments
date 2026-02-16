#!/usr/bin/env python3
"""Диагностика окружения внутри Docker: torch, transformers, PYTHONPATH."""
import sys
import os

print("=" * 60)
print("DIAGNOSTIC: Docker environment check")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print(f"sys.path (first 5): {sys.path[:5]}")
print()

# 1. Direct torch import
print("1. import torch")
try:
    import torch
    print(f"   OK: {torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()
print()

# 2. torch location
print("2. torch module location")
try:
    import torch
    print(f"   {torch.__file__}")
except Exception as e:
    print(f"   (torch not loaded: {e})")
print()

# 3. transformers import torch check
print("3. transformers.is_torch_available()")
try:
    from transformers.utils import is_torch_available
    print(f"   is_torch_available() = {is_torch_available()}")
except Exception as e:
    print(f"   FAIL: {e}")
print()

# 4. AutoModelForCausalLM
print("4. from transformers import AutoModelForCausalLM")
try:
    from transformers import AutoModelForCausalLM
    print("   OK")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()
print()

# 5. Check for shadowing - anything named torch in workspace?
print("5. Check workspace for torch shadowing")
workspace = "/workspace"
if os.path.exists(workspace):
    for name in ["torch", "torch.py"]:
        path = os.path.join(workspace, name)
        if os.path.exists(path):
            print(f"   WARNING: {path} exists - could shadow real torch!")
        else:
            print(f"   {path}: not found (OK)")
else:
    print("   /workspace not mounted")
print()

# 6. pip list torch
print("6. pip list | grep -i torch")
import subprocess
r = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "torch" in line.lower():
        print(f"   {line}")
print("=" * 60)
