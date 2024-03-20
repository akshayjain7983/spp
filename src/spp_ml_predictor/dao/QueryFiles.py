from .QueryHolder import readQueries1

SPP_STOCK_DATA_SQL = "/run/media/WORK/git_repos/spp/src/spp_ml_predictor/queries/stockData.sql"

def load():
    fileList = [SPP_STOCK_DATA_SQL]
    for e in fileList:
        readQueries1(e, '<<', '>>')

