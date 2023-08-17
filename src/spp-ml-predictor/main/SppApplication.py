from pyspark.sql import SparkSession
from ..dao import SppMLTrainingDao
from ..dao import SppMLTrainingDao
from ..dao import QueryFiles
from ..trainer import SppTrainer
import sys

print(sys.path)

def runSpp(ctx:dict):
    spark = (SparkSession
             .builder
             .remote("sc://localhost:15002")
            .appName("Spp")
            # .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
            # .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
            # .config("spark.mongodb.read.database", "spp")
            # .config("spark.mongodb.write.database", "spp")
             .getOrCreate())

    sppMLTrainingDao:SppMLTrainingDao = SppMLTrainingDao.SppMLTrainingDao(spark)
    sppTrainer:SppTrainer = SppTrainer.SppTrainer(sppMLTrainingDao, ctx)
    sppTrainer.train()




if __name__ == '__main__':

    QueryFiles.load()

    args = sys.argv[1:]
    trainingStartDate = args[0]
    trainingEndDate = args[1]
    exchange = args[2]
    index = args[3]
    exchangeCode = args[4] if len(args) == 5 else None

    ctx = {'exchange': exchange
            , 'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'index': index
            , 'exchangeCode': exchangeCode}

    runSpp(ctx)