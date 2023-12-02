from .QueryHolder import readQueries1

# SPP_STOCK_DATA_MQL = "/spp_ml_predictor/queries/stockData.mql"
SPP_STOCK_DATA_MQL = "/run/media/WORK/git_repos/spp/src/spp_ml_predictor/queries/stockData.mql"
SPP_STOCK_DATA_SPARK_MQL = "/run/media/WORK/git_repos/spp/src/spp_ml_predictor/queries/stockData-spark.mql"

def load():
    fileList = [SPP_STOCK_DATA_MQL, SPP_STOCK_DATA_SPARK_MQL]
    for e in fileList:
        readQueries1(e, '<', '>', 'CollectionName')

