from .QueryHolder import readQueries1

SPP_STOCK_DATA_MQL = "spp-ml-predictor/queries/stockData.mql"

def load():
    fileList = [SPP_STOCK_DATA_MQL]
    for e in fileList:
        readQueries1(e, '<', '>', 'CollectionName')

