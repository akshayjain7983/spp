from pyspark.sql import SparkSession
from .dao import SppMLTrainingDao
from .dao import QueryFiles
from .trainer import SppTrainer
import sys

def runSpp(ctx:dict):

    spark = None
    sparkMode = ctx['sparkMode']

    if(sparkMode == 'submit'):
        spark = (SparkSession
                 .builder
                 .appName("Spp")
                 .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
                 .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
                 .config("spark.mongodb.read.database", "spp")
                 .config("spark.mongodb.write.database", "spp")
                 .getOrCreate())
    else:
        spark = (SparkSession
                 .builder
                 .appName("Spp")
                 .remote("sc://localhost:15002")
                 .getOrCreate())

    sppMLTrainingDao:SppMLTrainingDao = SppMLTrainingDao.SppMLTrainingDao(spark)
    sppTrainer:SppTrainer = SppTrainer.SppTrainer(sppMLTrainingDao, ctx)
    sppTrainer.train()

def main(args):
    QueryFiles.load()

    trainingStartDate = args[0]
    trainingEndDate = args[1]
    exchange = args[2]
    index = args[3]
    exchangeCode = args[4] if len(args) >= 5 else None
    sparkMode = args[5] if len(args) >= 6 else 'connect'

    ctx = {'exchange': exchange
        , 'trainingStartDate': trainingStartDate
        , 'trainingEndDate': trainingEndDate
        , 'index': index
        , 'exchangeCode': exchangeCode
        , 'sparkMode': sparkMode}

    runSpp(ctx)


if __name__ == '__main__':
    main(sys.argv[1:])
