import numpy as np
import pandas as pd
import datetime as dt
import re

def get_stock_info(ticker_symbol = 3694):
    nikkei_url = 'https://www.nikkei.com/nkd/company/history/dprice/?scode={}&ba=1'.format(ticker_symbol)
    frame = pd.read_html(nikkei_url)[0]

    stock_values = np.array(frame['修正後終値'].tolist())[::-1]
    stock_dates = np.array(frame['日付'].tolist())[::-1]

    return stock_values, stock_dates

def extr_text(key, text):
    return re.findall(key, text)[0]

def shape_dates(dates):
    N = len(dates)

    dates_list = []
    m_d_list = []
    for i in range(N):
        m_d = '{}/{}'.format(extr_text('^\d+', dates[i]), extr_text('/(.*)（', dates[i]))
        m_d_list.append(m_d)
        dates_list.append(dt.datetime.strptime(m_d, '%m/%d')) # 年を考慮していない

    return m_d_list, dates_list

if __name__ == "__main__":
    get_stock_info(3694)
