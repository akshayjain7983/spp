import sys
import os
from pathlib import Path

path = os.path.dirname(__file__)
parent = Path(path).parent.absolute()
sys.path.insert(0, str(parent))
print(sys.path)