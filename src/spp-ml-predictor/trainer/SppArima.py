import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, date, timedelta
def buildModel(trainingDataPdf:pd.DataFrame) -> pd.DataFrame:
    trainingDataPdf.set_index("date", inplace=True)
    periodDays = 90
    model = ARIMA(trainingDataPdf.pScore, order=(periodDays, 2, 0))
    modelFit = model.fit()
    forcast = modelFit.get_forecast(periodDays)
    endDate = datetime.strftime(datetime.strptime(trainingDataPdf.index[len(trainingDataPdf.index)-1], '%Y-%m-%d') + timedelta(days=forcast.tvalues.keys()[periodDays-1]), '%Y-%m-%d')
    endPScore = forcast.tvalues[forcast.tvalues.keys()[periodDays-1]]
    return pd.DataFrame({"date":endDate, "pScore":endPScore}, index=[0])