import pandas as pd
import time
from ..trainer import SppArima
from statsmodels.tsa.arima.model import ARIMA
from pyspark.sql.functions import pandas_udf


class SppTrainingTask:
    def __init__(self, securityDataForExchangeCode:pd.DataFrame):
        self.securityDataForExchangeCode = securityDataForExchangeCode
        self.exchangeCode = securityDataForExchangeCode.head(1)['exchangeCode'][0]

    def buildModel(self) -> pd.DataFrame:

        startT = time.time();
        forecast = SppArima.buildModel(self.securityDataForExchangeCode)
        endT = time.time()
        print("Time taken:"+str(endT-startT)+" secs")
        print(forecast)
        return forecast