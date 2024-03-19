import pandas as pd
from datetime import datetime, date, timedelta
import pmdarima as pm
from ..trainer.SppForecaster import SppForecaster

class SppArima(SppForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppArima"

    def __getName__(self):
        return self.name
    def forecast(self) -> pd.DataFrame:

        forecastDays = self.ctx['forecastDays']
        trainingData = self.trainingDataPdf['value']
        model = pm.auto_arima(trainingData, X=self.xtraDataPdf[self.xtraDataPdf.index.isin(trainingData.index)]
                              , start_p=1, start_q=1,
                                test='adf',       # use adftest to find optimal 'd'
                                max_p=forecastDays[-1], max_q=forecastDays[-1], # maximum p and q
                                m=1,              # frequency of series
                                d=None,           # let model determine 'd'
                                seasonal=False,   # No Seasonality
                                start_P=0,
                                D=0,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
        forecast = model.predict(n_periods=forecastDays[-1], X=self.xtraDataPdf[-(forecastDays[-1]):])

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'D', "forecastDate": datetime.strftime(forecast.keys()[d - 1], '%Y-%m-%d'), "value": forecast[d - 1]} for d in forecastDays]
        }


        return pd.DataFrame(forecastValues, index=forecastDays)