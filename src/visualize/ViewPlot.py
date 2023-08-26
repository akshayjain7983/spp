import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

if __name__ == '__main__':
    mongoClient = MongoClient("mongodb://localhost:27017")
    forecastPScoreMql = {"exchangeCode":"500086"}
    forecastPScoreMongoResult = mongoClient['spp']['forecastPScore'].find(forecastPScoreMql)
    forecastPScorePd = pd.DataFrame(list(forecastPScoreMongoResult))
    forecastPScorePd = forecastPScorePd[['date', 'forecastedPScore']]
    forecastPScorePd.set_index('date', drop=False, inplace=True)
    forecastPScorePd.sort_index(inplace=True)

    trainingPScoreMql = {"exchangeCode": "500086", "date":{"$gte":"2021-12-31", "$lte":"2022-12-30"}}
    trainingPScoreMongoResult = mongoClient['spp']['trainingPScore'].find(trainingPScoreMql)
    trainingPScorePd = pd.DataFrame(list(trainingPScoreMongoResult))
    trainingPScorePd = trainingPScorePd[['date', 'trainingPScore']]
    trainingPScorePdSeries = [d.get('90D').get('pScore') for d in trainingPScorePd["trainingPScore"]]
    trainingPScorePd.drop("trainingPScore", axis=1, inplace=True)
    trainingPScorePd["trainingPScore"] = trainingPScorePdSeries
    trainingPScorePd.set_index("date", inplace=True, drop=False)
    trainingPScorePd.sort_index(inplace=True)

    forecastPScorePd[forecastPScorePd.index.isin(trainingPScorePd.index)]

    plt.rcParams['figure.figsize'] = [19.20, 10.80]
    plt.plot(forecastPScorePd['forecastedPScore'], label='forecast')
    plt.plot(trainingPScorePd['trainingPScore'], label='actual')
    # plt.ylim(-10, 10)
    plt.legend()

    # plt.show()
    plt.savefig("/run/media/WORK/MTechDissertation/spp-ml-models/500086-Arima-forecast-20211231-20221230.jpg", bbox_inches='tight')