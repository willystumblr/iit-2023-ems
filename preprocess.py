import glob
import pandas as pd
from tqdm import tqdm
from fix import fix_pvdata, fix_weather, PATH

for file in tqdm(glob.glob(PATH+"/*.csv")):
    data = pd.read_csv(file)
    building = file.split("/")[-1].split("-")[0]
    
    data = fix_pvdata(data)
    data = fix_weather(data, pd.Index(['적설(cm)', '풍속(m/s)', '풍향(16방위)', '외기온도(℃)', '강수량(mm)', '지면온도(°C)']))
    
    nan = data.isnull().sum().sum()
    #print(f"{building} has been preprocessed; NaN: {nan}")
    data.to_csv("./train/{}.csv".format(building), index=False)
