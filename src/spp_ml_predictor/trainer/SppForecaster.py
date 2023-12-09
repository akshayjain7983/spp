import pyspark.sql as ps

class SppForecaster:
    def __init__(self, trainingDataPdf:ps.DataFrame, ctx:dict, xtraDataPdf:ps.DataFrame):
        self.trainingDataPdf = trainingDataPdf
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def __getName__(self):
        return None

    def forecast(self) -> ps.DataFrame:
        pass