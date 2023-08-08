import pandas as pd

PATH = "/Users/minsk/2023ict/datasets"

def fix_weather(df: pd.DataFrame, to_remove: pd.Index) -> pd.DataFrame:
    # Remove unnecessary columns
    df = df.drop(columns=to_remove)
    for col in ['이슬점온도(°C)', '습도(%)', '증기압(hPa)', '시정(10m)']:
        ## ffill()
        df[col] = df[col].ffill()
        
    df['기온(°C)'] = df['기온(°C)'].bfill() # bfill()
    
    for col in ['일조(hr)', '일사(MJ/m2)', '전운량(10분위)']:
        # fillna(0.0)
        df[col] = df[col].fillna(0.0)
    
   
    for i in df[df['현지기압(hPa)'].isnull()].index:
        # the day before
        df['현지기압(hPa)'].iloc[i] = df['현지기압(hPa)'].iloc[i-24].copy()
        
    return df

def fix_pvdata(df: pd.DataFrame) -> pd.DataFrame:
    # Remove culmulative data
    df = df.drop(columns=['누적발전량(kWh)'])
    
    # datatype
    for i in df[df['시간당발전량(kWh)']=='-'].index:
        # the day before
        df['시간당발전량(kWh)'].iloc[i] = df['시간당발전량(kWh)'].iloc[i-24]
        
    df['시간당발전량(kWh)'] = df['시간당발전량(kWh)'].astype('float64')
    
    # create hour and month for categorical data
    df['일시'] = pd.to_datetime(df['일시'])
    df['HOUR'] = df['일시'].dt.hour
    df['MONTH'] = df['일시'].dt.month
    df['DAY'] = df['일시'].dt.day
    df['YEAR'] = df['일시'].dt.year
    df = df.drop(columns=['일시'])
    
    # fix '경사면(w/㎡)'
    abnormals = df[(df['경사면(w/㎡)'] > 2500)].index
    print(abnormals)
    for i in abnormals:
        df['경사면(w/㎡)'].iloc[i]=df['경사면(w/㎡)'].iloc[i-1]
    
    #fix '모듈온도(℃)'
    abnormals = df[(df['모듈온도(℃)'] < -500)].index
    print(abnormals)
    for i in abnormals:
        df['모듈온도(℃)'].iloc[i]=df['모듈온도(℃)'].iloc[i-1]
    
    return df


