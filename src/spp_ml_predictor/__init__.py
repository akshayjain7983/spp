import sys
import os
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(encoding='utf-8', level=logging.DEBUG, format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')
logger = logging.getLogger('spp')

pd.options.mode.chained_assignment = None
path = os.path.dirname(__file__)
parent = Path(path).parent.absolute()
sys.path.insert(0, str(parent))
logging.info(sys.path)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"