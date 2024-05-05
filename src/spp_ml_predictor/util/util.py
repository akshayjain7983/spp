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

def candlestick_movement(open:float, close:float):
    return (close / open - 1) if(open != 0.0) else 0.0

def candlestick_real_body_change(open:float, close:float):
    return (close / open) if(open != 0.0) else 0.0

def candlestick_full_body_change(low:float, high:float):
    return (high / low) if(low != 0.0) else 0.0