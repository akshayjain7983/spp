import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    forecastPeriod = [10]
    mongoClient = MongoClient("mongodb://localhost:27017")
    forecastModel = "SppNN"
    exchangeCode = "509488"
    dateStart = "2018-09-01"
    dateEnd = "2019-08-31"

    organizationMql = {"exchangeCode":exchangeCode}
    organization = mongoClient['spp']['organizations'].find_one(organizationMql)

    forecastPScoreMql = {"exchangeCode":exchangeCode, "date":{"$gte":dateStart, "$lte":dateEnd}, "forecast.forecastModel":forecastModel}
    forecastPScoreMongoResult = mongoClient['spp']['forecastPScore'].find(forecastPScoreMql)
    forecastPScorePd = pd.DataFrame(list(forecastPScoreMongoResult))

    trainingPScoreMql = {"exchangeCode": exchangeCode, "date": {"$gte": dateStart, "$lte": dateEnd}}
    trainingPScoreMongoResult = mongoClient['spp']['trainingPScore'].find(trainingPScoreMql)
    trainingPScorePd = pd.DataFrame(list(trainingPScoreMongoResult))

    plt.rcParams['figure.figsize'] = [19.20, 10.80]
    plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 95, "Stock = {} ({} - {})".format(organization['name'], organization['exchange'], organization['exchangeCode']), fontsize=10)
    plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 90, "Forecast Model = {}".format(forecastModel), fontsize=10)

    for c, f in enumerate(forecastPeriod):
        fp = str(f)+"D"
        forecastPScorePdForPeriod = forecastPScorePd.copy()
        forecastPScorePdForPeriodSeries = [d.get(fp).get('forecastedPScore') for d in forecastPScorePdForPeriod["forecast"]]
        forecastIndexReturnPdForPeriodSeries = [d.get(fp).get('forecastedIndexReturn') for d in forecastPScorePdForPeriod["forecast"]]
        forecastSecurityReturnPdForPeriodSeries = [d.get(fp).get('forecastedSecurityReturn') for d in forecastPScorePdForPeriod["forecast"]]
        forecastPScorePdForPeriod["forecastedPScore"] = [v for v in forecastPScorePdForPeriodSeries]
        forecastPScorePdForPeriod["forecastedIndexReturn"] = [v*100 for v in forecastIndexReturnPdForPeriodSeries]
        forecastPScorePdForPeriod["forecastedSecurityReturn"] = [v*100 for v in forecastSecurityReturnPdForPeriodSeries]
        forecastPScorePdForPeriod['datetime'] = pd.to_datetime(forecastPScorePdForPeriod['date'])
        forecastPScorePdForPeriod.set_index('datetime', drop=True, inplace=True)
        forecastPScorePdForPeriod.sort_index(inplace=True)
        forecastPScorePdForPeriod.drop_duplicates(subset=['date'], inplace=True)

        trainingPScorePdForPeriod = trainingPScorePd.copy()
        trainingPScorePdForPeriod = trainingPScorePdForPeriod[['date', 'trainingPScore']]
        trainingPScorePdForPeriodSeries = [d.get(fp).get('pScore') for d in trainingPScorePdForPeriod["trainingPScore"]]
        trainingIndexReturnPdForPeriodSeries = [d.get(fp).get('indexReturns') for d in trainingPScorePdForPeriod["trainingPScore"]]
        trainingSecurityReturnPdForPeriodSeries = [d.get(fp).get('securityReturns') for d in trainingPScorePdForPeriod["trainingPScore"]]
        trainingPScorePdForPeriod.drop("trainingPScore", axis=1, inplace=True)
        trainingPScorePdForPeriod["trainingPScore"] = trainingPScorePdForPeriodSeries
        trainingPScorePdForPeriod["indexReturns"] = [v*100 for v in trainingIndexReturnPdForPeriodSeries]
        trainingPScorePdForPeriod["securityReturns"] = [v*100 for v in trainingSecurityReturnPdForPeriodSeries]
        trainingPScorePdForPeriod['datetime'] = pd.to_datetime(trainingPScorePdForPeriod['date'])
        trainingPScorePdForPeriod.set_index('datetime', drop=True, inplace=True)
        trainingPScorePdForPeriod.sort_index(inplace=True)

        forecastPScorePdForPeriod[forecastPScorePdForPeriod.index.isin(trainingPScorePdForPeriod.index)]

        rmsePscore = sqrt(mean_squared_error(trainingPScorePdForPeriod['trainingPScore'], forecastPScorePdForPeriod['forecastedPScore']))
        rmseIndexReturns = sqrt(mean_squared_error(trainingPScorePdForPeriod['indexReturns'], forecastPScorePdForPeriod['forecastedIndexReturn']))
        rmseSecurityReturns = sqrt(mean_squared_error(trainingPScorePdForPeriod['securityReturns'], forecastPScorePdForPeriod['forecastedSecurityReturn']))

        plt.plot(forecastPScorePdForPeriod['forecastedPScore'], label=('forecastedPScore'+fp))
        # plt.plot(forecastPScorePdForPeriod['forecastedIndexReturn'], label=('forecastedIndexReturn'+fp))
        # plt.plot(forecastPScorePdForPeriod['forecastedSecurityReturn'], label=('forecastedSecurityReturn'+fp))
        plt.plot(trainingPScorePdForPeriod['trainingPScore'], label=('actualPScore'+fp))
        # plt.plot(trainingPScorePdForPeriod['indexReturns'], label=('actualIndexReturns'+fp))
        # plt.plot(trainingPScorePdForPeriod['securityReturns'], label=('actualSecurityReturns'+fp))
        y = 85 if c == 0 else 70 if c == 1 else 55
        plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), y, "RMSE PScore {} = {}".format(fp, str(rmsePscore)), fontsize = 10)
        # plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), (y-5), "RMSE Index Returns {} = {}".format(fp, str(rmseIndexReturns)), fontsize = 10)
        # plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), (y-10), "RMSE Security Returns {} = {}".format(fp, str(rmseSecurityReturns)), fontsize = 10)

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("PScore", fontsize=15)
    plt.ylim(-100, 100)
    plt.legend()

    fileName = "SppForecasting_{}_{}_{}_{}_{}.jpg".format(exchangeCode, dateStart, dateEnd, forecastModel, str(forecastPeriod[0]))
    path = "/run/media/WORK/MTechDissertation/spp-ml-models/"
    plt.savefig(path+fileName, bbox_inches='tight')
    plt.show()
