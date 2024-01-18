import joblib
import requests
import time
import pandas as pd
import get_label_indicator
from dataDogutils import DataDogUtils

pd.set_option('expand_frame_repr',False)

def get_10s_trades(start_time,end_time):
    BASE_URL = 'https://api.binance.com'
    trades_url = BASE_URL + '/api/v3/aggTrades' + '?' +'symbol=BTCUSDT&startTime='+str(start_time)+'&endTime='+str(end_time)
    #print(trades_url)
    trades = pd.DataFrame(requests.get(trades_url).json())
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
    df_21min = pd.DataFrame()
    end_10s_time = time.time() //10 * 10 * 1000
    for i in range(126):
        trades_agg = pd.DataFrame(get_10s_trades(int(end_10s_time-(i+1)*10000),int(end_10s_time-i*10000)))
        df_21min = pd.concat([trades_agg,df_21min])
    model_all = joblib.load("/data/market_maker/model/model_all_1pct_20240107.pkl")
    model_volumn_all = joblib.load("/data/market_maker/model/model_volumn_all_1pct_20240107.pkl")
    model_buy_all = joblib.load("/data/market_maker/model/model_buy_all_1pct_20240107.pkl")
    model_sell_all = joblib.load("/data/market_maker/model/model_sell_all_1pct_20240107.pkl")
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
            print(f"UTC{pd.to_datetime(seconds+10, unit = 's')}——{pd.to_datetime(seconds+20, unit= 's')}  alert:")
            if all_pred == 1:
                print('all_pred   ')
            if volumn_all_pred == 1:
                print('volumn_all_pred   ')
            if buy_all_pred == 1:
                print('buy_all_pred   ')
            if sell_all_pred == 1:
                print('sell_all_pred')

            DataDogUtils.gauge(
                "volumn_sell_buy_alert",
                all_pred,
                tags=[f"time:{pd.to_datetime(seconds+10, unit = 's')}"],
            )
            DataDogUtils.gauge(
                "volumn_alert",
                volumn_all_pred,
                tags=[f"time:{pd.to_datetime(seconds+10, unit = 's')}"],
            )
            DataDogUtils.gauge(
                "sell_alert",
                sell_all_pred,
                tags=[f"time:{pd.to_datetime(seconds+10, unit = 's')}"],
            )
            DataDogUtils.gauge(
                "buy_alert",
                buy_all_pred,
                tags=[f"time:{pd.to_datetime(seconds+10, unit = 's')}"],
            )

        time.sleep(1)
