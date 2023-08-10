import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer

# Define a custom PyTorch dataset
class SolarDataset(Dataset):
    def __init__(self, dataframe, seq_len=7*24):
        self.seq_len = seq_len
        self.sequences, self.labels = self._preprocess(dataframe)
        
        
        
    def _preprocess(self, dataframe):
        scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        # Separate numerical and categorical features
        numerical_features = ['시간당발전량(kWh)', '수평면(w/㎡)', '경사면(w/㎡)', '모듈온도(℃)', '기온(°C)', '습도(%)',
                           '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '일조(hr)',
                           '일사(MJ/m2)', '시정(10m)']
        categorical_features = ['전운량(10분위)']
        cyclic_features = ['Hour', 'Month']
        cyclic_preprocessed = ['Hour_sin', 'Hour_cos','Month_sin', 'Month_cos']
        
        # Create a Column Transformer to handle numerical and categorical columns
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('cat', 'passthrough', categorical_features),
            ('cyc', 'passthrough', cyclic_preprocessed)
        ])
        
        
        # Perform cyclical encoding for cyclic features
        def cyclical_encode(df, feature, max_val):
            df[feature + '_sin'] = np.sin(2 * np.pi * df[feature] / max_val)
            df[feature + '_cos'] = np.cos(2 * np.pi * df[feature] / max_val)
            df = df.drop(feature, axis=1)
            return df
        
        # Apply cyclical encoding for each cyclic feature
        for feature in cyclic_features:
            max_val = dataframe[feature].max()
            dataframe = cyclical_encode(dataframe, feature, max_val)

        X_preprocessed = preprocessor.fit_transform(dataframe)
        y_preprocessed = y_scaler.fit_transform(dataframe['시간당발전량(kWh)'].to_numpy().reshape(-1,1))
        
        # Convert the preprocessed data into a PyTorch tensor
        X_tensor = torch.tensor(X_preprocessed, dtype=torch.float32)
        y_tensor = torch.tensor(y_preprocessed, dtype=torch.float32)
        
        # Create sequences and labels
        
        sequences = []
        labels = []
        
        if (len(X_tensor) - self.seq_len - 24)==0:
            sequences.append(X_tensor[:self.seq_len])
            labels.append(y_tensor[self.seq_len:self.seq_len+24])
        else:
            for i in range(len(X_tensor) - self.seq_len - 24):  # Subtract 24 to account for predicting the next 24 hours
                seq = X_tensor[i:i+self.seq_len]
                label = y_tensor[i+self.seq_len:i+self.seq_len+24]  # Next 24 hours
                sequences.append(seq)
                labels.append(label)
        
        return torch.stack(sequences), torch.stack(labels)
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx].squeeze()


# 시계열 데이터를 처리하는 클래스를 정의합니다.
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, seq_len=7*24, pred_len=24):
        self.seq_len = seq_len  # 입력 시퀀스의 길이를 정의합니다.
        self.pred_len = pred_len  # 예측할 시퀀스의 길이를 정의합니다.
        self.scaler = MinMaxScaler()  # 데이터 정규화를 위한 MinMaxScaler 객체를 생성합니다.

        self.dataframe = self._preprocess(dataframe)  # 데이터 전처리 함수를 호출하여 dataframe을 전처리합니다.

    def _preprocess(self, df):
        # 누락된 값을 시계열의 이전 값으로 채웁니다.
        df.fillna(method='ffill', inplace=True)

        # 숫자형 열을 [0, 1] 범위로 정규화합니다.
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # 범주형 변수를 원-핫 인코딩합니다.
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            encoder = OneHotEncoder()
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
            
            # 원래의 범주형 열을 삭제하고 인코딩된 열과 병합합니다.
            df.drop(columns=categorical_cols, inplace=True)
            df = pd.concat([df, encoded_df], axis=1)
        
        return df

    def __len__(self):
        return len(self.dataframe) - self.seq_len - self.pred_len + 1  # 데이터셋의 전체 길이를 반환합니다.

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx:idx+self.seq_len, :7]  # 입력 시퀀스의 앞 7열만 선택합니다.
        # 마지막 56열이 전력 값이라고 가정하고 예측할 시퀀스를 선택합니다.
        y = self.dataframe.iloc[idx+self.seq_len:idx+self.seq_len+self.pred_len, -56:] 
        return torch.Tensor(x.values), torch.Tensor(y.values).reshape(-1)  # y 값을 평탄화하여 반환합니다.