import joblib
import requests
import time
import pandas as pd
import get_label_indicator
from dataDogutils import DataDogUtils
from requests.exceptions import ConnectionError
import logging
import sys

# 配置日志记录
logging.basicConfig(filename='volumn_predictor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_10s_trades(start_time,end_time):
    BASE_URL = 'https://api-gcp.binance.com'
    trades_url = BASE_URL + '/api/v3/aggTrades' + '?' +'symbol=BTCUSDT&startTime='+str(start_time)+'&endTime='+str(end_time)
    #print(trades_url)

    # 设置重试次数
    max_retries = 300
    retries = 0
    # 尝试连接并捕获异常
    while retries < max_retries:
        try:
            print(requests.get(trades_url).json())
            trades = pd.DataFrame(requests.get(trades_url).json())
            break  # 如果成功则跳出循环

        except ConnectionError as e:  # 捕捉特定的 ConnectionError
            print(f"ConnectionError异常:{e}")
            logging.info(f"ConnectionError异常:{e}")
            if retries < max_retries - 1:
                logging.info(f"进行第 {retries+1} 次重试...")
                print(f"进行第 {retries+1} 次重试...")
                time.sleep(10)  # 等待10秒
            retries += 1
    else:
        logging.info("重试达到最大次数，放弃重试")
        print("重试达到最大次数，放弃重试")
        sys.exit(1)
    trades = trades.apply(pd.to_numeric, errors='coerce')
    trades['r'] = trades['p'] * trades['q']
    df_agg = pd.DataFrame(columns = ['t','open','close','max','min','sell','sell_r','sell_count','buy','buy_r','buy_count'])
    df_agg.loc[0,'t'] = pd.to_datetime(start_time,unit='ms')
    df_agg.loc[0,'open'] = trades['p'].iloc[0]
    df_agg.loc[0,'close'] = trades['p'].iloc[-1]
    df_agg.loc[0,'max'] = trades['p'].max()
    df_agg.loc[0,'min'] = trades['p'].min()
    df_agg.loc[0,'sell'] = trades[trades['m'] == True]['q'].sum()
    df_agg.loc[0,'sell_r'] = trades[trades['m'] == True]['r'].sum()
    df_agg.loc[0,'sell_count'] = (trades[trades['m'] == True]['l']-trades[trades['m'] == True]['f']+1).sum()
    df_agg.loc[0,'buy'] = trades[trades['m'] == False]['q'].sum()
    df_agg.loc[0,'buy_r'] = trades[trades['m'] == False]['r'].sum()
    df_agg.loc[0,'buy_count'] = (trades[trades['m'] == False]['l']-trades[trades['m'] == False]['f']+1).sum()
    df_agg.index = pd.to_datetime(df_agg['t'])
    df_agg.drop(columns = ['t'],axis = 1,inplace = True)

    
    return df_agg
    
    

if __name__ == '__main__':
    logging.info('程序开始运行')
    df_21min = pd.DataFrame()
    end_10s_time = time.time() //10 * 10 * 1000
    model_all = joblib.load("/data/market_maker/model/model_all_1pct_20240107.pkl")
    model_volumn_all = joblib.load("/data/market_maker/model/model_volumn_all_1pct_20240107.pkl")
    model_buy_all = joblib.load("/data/market_maker/model/model_buy_all_1pct_20240107.pkl")
    model_sell_all = joblib.load("/data/market_maker/model/model_sell_all_1pct_20240107.pkl")
    for i in range(126):
        #time.sleep(0.1)
        trades_agg = pd.DataFrame(get_10s_trades(int(end_10s_time-(i+1)*10000),int(end_10s_time-i*10000)))
        df_21min = pd.concat([trades_agg,df_21min])
    while True:
        # 获取最近十秒的历史交易数据
        seconds = time.time() // 1
        if seconds % 10 == 0:
            recent_trades_agg = get_10s_trades(int(seconds // 10 * 10 * 1000-10000), int(seconds //10 * 10 * 1000))
            df_21min = pd.concat([df_21min[1:],recent_trades_agg])
            df_indicator = get_label_indicator.get_indicators(df_21min)
            # 进行分类预测
            all_pred = int(model_all.predict(df_indicator.drop(columns = ['all_1pct'],axis = 1,inplace = False).iloc[[-1]]))
            volumn_all_pred = int(model_volumn_all.predict(df_indicator.drop(columns = ['volumn_all_1pct'],axis = 1,inplace = False).iloc[[-1]]))
            buy_all_pred = int(model_buy_all.predict(df_indicator.drop(columns = ['buy_all_1pct'],axis = 1,inplace = False).iloc[[-1]]))
            sell_all_pred = int(model_sell_all.predict(df_indicator.drop(columns = ['sell_all_1pct'],axis = 1,inplace = False).iloc[[-1]]))
            #if (all_pred == 1)| (volumn_all_pred == 1) | (buy_all_pred == 1) | (sell_all_pred == 1):
            logging.info(f"UTC{pd.to_datetime(seconds+10, unit = 's')}——{pd.to_datetime(seconds+20, unit= 's')}  alert:")
            print(f"UTC{pd.to_datetime(seconds+10, unit = 's')}——{pd.to_datetime(seconds+20, unit= 's')}  alert:")
            if all_pred == 1:
                logging.info('all_pred   ')
                print('all_pred   ')
            if volumn_all_pred == 1:
                logging.info('volumn_all_pred   ')
                print('volumn_all_pred   ')
            if buy_all_pred == 1:
                logging.info('buy_all_pred   ')
                print('buy_all_pred   ')
            if sell_all_pred == 1:
                logging.info('sell_all_pred   ')
                print('sell_all_pred')

            DataDogUtils.gauge(
                "volumn_sell_buy_alert",
                all_pred,
            )
            DataDogUtils.gauge(
                "volumn_alert",
                volumn_all_pred,
            )
            DataDogUtils.gauge(
                "sell_alert",
                sell_all_pred,
            )
            DataDogUtils.gauge(
                "buy_alert",
                buy_all_pred,
            )

        time.sleep(1)
