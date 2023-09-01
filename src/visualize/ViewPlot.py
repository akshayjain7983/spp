import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    mongoClient = MongoClient("mongodb://localhost:27017")
    forecastPScoreMql = {"exchangeCode":"500086", "date":{"$gte":"2018-09-01", "$lte":"2019-08-31"}, "forecastPeriod":"30D", "lastUpdatedTimestamp":{"$gte":"2023-09-01"}, "forecastModel":"SppArima"}
    forecastPScoreMongoResult = mongoClient['spp']['forecastPScore'].find(forecastPScoreMql)
    forecastPScorePd = pd.DataFrame(list(forecastPScoreMongoResult))
    forecastPScorePd = forecastPScorePd[['date', 'forecastedPScore', 'forecastedIndexReturn', 'forecastedSecurityReturn']]
    forecastPScorePd["forecastedPScore"] = forecastPScorePd["forecastedPScore"]
    forecastPScorePd['datetime'] = pd.to_datetime(forecastPScorePd['date'])
    forecastPScorePd.set_index('datetime', drop=True, inplace=True)
    forecastPScorePd.sort_index(inplace=True)
    forecastPScorePdReindex = pd.date_range(start=datetime.strptime("2018-09-01", "%Y-%m-%d"), end=datetime.strptime("2019-08-31", "%Y-%m-%d"), inclusive="both")
    forecastPScorePd.drop_duplicates(subset=['date'], inplace=True)
    forecastPScorePd = forecastPScorePd.reindex(forecastPScorePdReindex)

    trainingPScoreMql = {"exchangeCode": "500086", "date":{"$gte":"2018-09-01", "$lte":"2019-08-31"}}
    trainingPScoreMongoResult = mongoClient['spp']['trainingPScore'].find(trainingPScoreMql)
    trainingPScorePd = pd.DataFrame(list(trainingPScoreMongoResult))
    trainingPScorePd = trainingPScorePd[['date', 'trainingPScore']]
    trainingPScorePdSeries = [d.get('30D').get('pScore') for d in trainingPScorePd["trainingPScore"]]
    trainingIndexReturnsPdSeries = [d.get('30D').get('indexReturns') for d in trainingPScorePd["trainingPScore"]]
    trainingSecurityReturnsPdSeries = [d.get('30D').get('securityReturns') for d in trainingPScorePd["trainingPScore"]]
    trainingPScorePd.drop("trainingPScore", axis=1, inplace=True)
    trainingPScorePd["trainingPScore"] = trainingPScorePdSeries
    trainingPScorePd["trainingPScore"] = trainingPScorePd["trainingPScore"]
    trainingPScorePd["indexReturns"] = trainingIndexReturnsPdSeries
    trainingPScorePd["securityReturns"] = trainingSecurityReturnsPdSeries
    trainingPScorePd['datetime'] = pd.to_datetime(trainingPScorePd['date'])
    trainingPScorePd.set_index('datetime', drop=True, inplace=True)
    trainingPScorePd.sort_index(inplace=True)

    forecastPScorePd[forecastPScorePd.index.isin(trainingPScorePd.index)]

    rmsePscore = sqrt(mean_squared_error(trainingPScorePd['trainingPScore'], forecastPScorePd['forecastedPScore']))
    rmseIndexReturns = sqrt(mean_squared_error(trainingPScorePd['indexReturns'], forecastPScorePd['forecastedIndexReturn']))
    rmseSecurityReturns = sqrt(mean_squared_error(trainingPScorePd['securityReturns'], forecastPScorePd['forecastedSecurityReturn']))
    print("RMSE PScore = "+str(rmsePscore))
    print("RMSE index returns = " + str(rmseIndexReturns))
    print("RMSE security returns = " + str(rmseSecurityReturns))

    plt.rcParams['figure.figsize'] = [19.20, 10.80]
    plt.plot(forecastPScorePd['forecastedPScore'], label='forecastedPScore')
    # plt.plot(forecastPScorePd['forecastedIndexReturn'], label='forecastedIndexReturn')
    # plt.plot(forecastPScorePd['forecastedSecurityReturn'], label='forecastedSecurityReturn')
    plt.plot(trainingPScorePd['trainingPScore'], label='actualPScore')
    # plt.plot(trainingPScorePd['indexReturns'], label='actualIndexReturns')
    # plt.plot(trainingPScorePd['securityReturns'], label='actualSecurityReturns')
    # plt.ylim(-10, 10)
    plt.legend()

    plt.show()
    # plt.savefig("/run/media/WORK/MTechDissertation/spp-ml-models/500086-Arima-forecast-20211231-20221230.jpg", bbox_inches='tight')