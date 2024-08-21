import time
from typing import Any

def _print(message:Any):
    print(f"[{time.ctime()}] {message}")