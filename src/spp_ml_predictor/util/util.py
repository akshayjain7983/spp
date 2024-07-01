from datetime import datetime, timedelta, date
import pandas as pd

def is_holiday(dateToCheck: date, holidays:pd.DataFrame):
    return holidays.at[dateToCheck, 'is_holiday']

def next_business_date(startDate: date, days:int, holidays:pd.DataFrame):
    businessDate = startDate
    count = 0
    while(count < days):
        businessDate = businessDate + timedelta(days=1)
        if(not is_holiday(businessDate, holidays)):
            count += 1

    return businessDate


def previous_business_date(endDate: date, days: int, holidays: pd.DataFrame):
    businessDate = endDate
    count = 0
    while (count < days):
        businessDate = businessDate - timedelta(days=1)
        if (not is_holiday(businessDate, holidays)):
            count += 1

    return businessDate

def business_date_range(startDate: date, endDate: date, holidays: pd.DataFrame):
    businessDates = []
    currentDate = startDate-timedelta(days=1) #start from a day before so that startDate is also tested
    while(currentDate <= endDate):
        currentDate = next_business_date(currentDate, 1, holidays)
        businessDates.append(currentDate)

    return businessDates

def candlestick_movement(open:float, close:float):
    return (close / open - 1) if(open != 0.0) else 0.0

def candlestick_real_body_change(open:float, close:float):
    return (close / open) if(open != 0.0) else 0.0

def candlestick_full_body_change(low:float, high:float):
    return (high / low) if(low != 0.0) else 0.0

def transformIndexData(trainingData:pd.DataFrame):
    indexLevelsPdf:pd.DataFrame = trainingData['indexLevelsPdf']
    indexLevelsPdf['datetime'] = pd.to_datetime(indexLevelsPdf['date'])
    indexLevelsPdf.set_index("datetime", inplace=True, drop=True)
    indexLevelsPdf.sort_index(inplace=True)
    trainingData['indexLevelsPdf'] = indexLevelsPdf

def transformSecurityData(trainingData:pd.DataFrame):
    securityPricesPdf:pd.DataFrame = trainingData['securityPricesPdf']
    securityPricesPdf['datetime'] = pd.to_datetime(securityPricesPdf['date'])
    securityPricesPdf.set_index("datetime", inplace=True, drop=True)
    securityPricesPdf.sort_index(inplace=True)
    trainingData['securityPricesPdf'] = securityPricesPdf

def candlestick_movement_apply_df(row):
    return candlestick_movement(row['open'], row['close'])

def determineIndexCandlestickPatterns(trainingData:pd.DataFrame):
    indexLevelsPdf: pd.DataFrame = trainingData['indexLevelsPdf']
    indexLevelsPdf['candlestickMovement'] = indexLevelsPdf.apply(candlestick_movement_apply_df, axis=1)

def determineSecurityCandlestickPatterns(trainingData:pd.DataFrame):
    securityPricesPdf: pd.DataFrame = trainingData['securityPricesPdf']
    securityPricesPdf['candlestickMovement'] = securityPricesPdf.apply(candlestick_movement_apply_df, axis=1)


def transformForIndex(trainingData:pd.DataFrame):
    transformIndexData(trainingData)
    determineIndexCandlestickPatterns(trainingData)
    return trainingData

def transformForSecurity(trainingData:pd.DataFrame):
    transformSecurityData(trainingData)
    determineSecurityCandlestickPatterns(trainingData)
    return trainingData