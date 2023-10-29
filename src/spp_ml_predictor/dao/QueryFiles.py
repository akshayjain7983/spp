from .QueryHolder import readQueries1

# SPP_STOCK_DATA_MQL = "/spp_ml_predictor/queries/stockData.mql"
SPP_STOCK_DATA_MQL = "/run/media/WORK/git_repos/SPP-ML-Predictor/src/spp_ml_predictor/queries/stockData.mql"

def load():
    fileList = [SPP_STOCK_DATA_MQL]
    for e in fileList:
        readQueries1(e, '<', '>', 'CollectionName')

