import sys
import os
from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None
path = os.path.dirname(__file__)
parent = Path(path).parent.absolute()
sys.path.insert(0, str(parent))
print(sys.path)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"