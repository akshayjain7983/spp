import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, date, timedelta
import pmdarima as pm

def buildModel(trainingDataPdf: pd.DataFrame, ctx:dict) -> pd.DataFrame:

    pScoreDate = ctx['pScoreDate']
    forecastDays = ctx['forecastDays']
    trainingData = trainingDataPdf['value']
    forecastValues = []
    # for i in range(forecastDays):
        # model = ARIMA(trainingData, order=(0, 1, 0))
    model = pm.auto_arima(trainingData, start_p=1, start_q=1,
                  test='adf',       # use adftest to find optimal 'd'
                  max_p=forecastDays, max_q=forecastDays, # maximum p and q
                  m=1,              # frequency of series
                  d=None,           # let model determine 'd'
                  seasonal=False,   # No Seasonality
                  start_P=0,
                  D=0,
                  trace=True,
                  error_action='ignore',
                  suppress_warnings=True,
                  stepwise=True)
    forecast = model.predict(n_periods=forecastDays)
    value = forecast.values[0]
    forecastValues.append(value)
    # trainingData._set_value(datetime.strftime(datetime.strptime(pScoreDate, '%Y-%m-%d') + timedelta(days=(i+1)), '%Y-%m-%d'), value)

    endDate = datetime.strftime(datetime.strptime(pScoreDate, '%Y-%m-%d') + timedelta(days=forecastDays), '%Y-%m-%d')
    return pd.DataFrame({"forecastDate": endDate, "value": forecastValues[len(forecastValues)-1], "forecastModel": "SppArima"}, index=[0])
