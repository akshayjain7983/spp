import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, date, timedelta


def buildModel(trainingDataPdf: pd.DataFrame) -> pd.DataFrame:
    periodDays = 90
    model = ARIMA(trainingDataPdf['pScore'], order=(periodDays, 1, 0))
    modelFit = model.fit()
    forcast = modelFit.get_forecast(periodDays)
    endDate = datetime.strftime(forcast.tvalues.keys()[periodDays - 1], '%Y-%m-%d')
    endPScore = forcast.tvalues[forcast.tvalues.keys()[periodDays - 1]]
    return pd.DataFrame({"date": endDate, "forecastPScore": endPScore, "forecastPScoreModel": "SppArima"}, index=[0])
