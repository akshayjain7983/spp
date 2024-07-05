import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
from math import sqrt
from spp_ml_predictor.config.ConfigReader import readConfig
# from ..spp_ml_predictor.config.ConfigReader import readConfig
from spp_ml_predictor.dao.SppMLDao import SppMLDao
from spp_ml_predictor.dao import QueryFiles

if __name__ == '__main__':
    QueryFiles.load()
    forecastPeriod = '10d'
    config = readConfig(configFilename='spp_ml_predictor/config.ini')
    sppMLTrainingDao: SppMLDao = SppMLDao(config)
    forecastModel = "SppNN"
    exchangeCode = "505200"
    dateStart = "2018-09-01"
    dateEnd = "2019-08-31"
    compare = 'pscore'

    ctx = {
        'startDate': dateStart
        , 'endDate': dateEnd
        , 'exchange': 'BSE'
        , 'index': 'SENSEX'
        , 'exchangeCodes': [exchangeCode]
        , 'forecastModel': forecastModel
        , 'forecastPeriod': forecastPeriod
        , 'scorePeriod': forecastPeriod
    }

    securitiesPd = sppMLTrainingDao.loadSecurityExchangeCodes(ctx)
    forecastPScorePd = sppMLTrainingDao.loadForecastPScore(ctx)
    forecastPScorePd.sort_index(inplace=True)
    actualPScorePd = sppMLTrainingDao.loadActualPScore(ctx)
    actualPScorePd.sort_index(inplace=True)

    plt.rcParams['figure.figsize'] = [19.20, 10.80]
    plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 95, "Stock = {} ({} - {})".format(securitiesPd['security_name'][0], securitiesPd['exchange'][0], securitiesPd['exchange_code'][0]), fontsize=10)
    plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 90, "Forecast Model = {}".format(forecastModel), fontsize=10)

    if(compare == 'pscore'):
        plt.plot(forecastPScorePd['forecasted_p_score'], label=('forecastedPScore-' + forecastPeriod))
        plt.plot(actualPScorePd['calculated_p_score'], label=('actualPScore-' + forecastPeriod))
        mapePscore = mean_absolute_percentage_error(actualPScorePd['calculated_p_score'], forecastPScorePd['forecasted_p_score'])
        rmsePscore = root_mean_squared_error(actualPScorePd['calculated_p_score'], forecastPScorePd['forecasted_p_score'])
        plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 85, "RMSE PScore {} = {}".format(forecastPeriod, str(rmsePscore)), fontsize=10)
        plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 80, "MAPE PScore {} = {}".format(forecastPeriod, str(mapePscore)), fontsize=10)

    if(compare == 'index_return'):
        plt.plot(forecastPScorePd['forecasted_index_return'] * 100, label=('forecastedIndexReturn-' + forecastPeriod))
        plt.plot(actualPScorePd['actual_index_return'] * 100, label=('actualIndexReturns-' + forecastPeriod))
        mapeIndexReturns = mean_absolute_percentage_error(actualPScorePd['actual_index_return'], forecastPScorePd['forecasted_index_return'])
        plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 80, "MAPE Index Returns {} = {}".format(forecastPeriod, str(mapeIndexReturns)), fontsize=10)

    if (compare == 'security_return'):
        plt.plot(forecastPScorePd['forecasted_security_return'] * 100, label=('forecastedSecurityReturn-' + forecastPeriod))
        plt.plot(actualPScorePd['actual_security_return'] * 100, label=('actualSecurityReturns-' + forecastPeriod))
        mapeSecurityReturns = mean_absolute_percentage_error(actualPScorePd['actual_security_return'], forecastPScorePd['forecasted_security_return'])
        plt.text(datetime.strptime(dateStart, '%Y-%m-%d'), 75, "MAPE Security Returns {} = {}".format(forecastPeriod, str(mapeSecurityReturns)), fontsize = 10)

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("PScore", fontsize=15)
    plt.ylim(-100, 100)
    plt.legend()

    # fileName = "SppForecasting_{}_{}_{}_{}_{}.jpg".format(exchangeCode, dateStart, dateEnd, forecastModel, str(forecastPeriod[0]))
    # path = "/run/media/WORK/MTechDissertation/spp-ml-models/"
    # plt.savefig(path+fileName, bbox_inches='tight')
    plt.show()
