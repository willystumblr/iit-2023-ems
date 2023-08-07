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


class TimeSeriesDataset_forPredict(Dataset):
    def __init__(self, dataframe, seq_len=7*24):
        self.seq_len = seq_len

        self.dataframe = self._preprocess(dataframe)

    def _preprocess(self, df):
        # If there are any missing values, fill them with the previous value in time-series
        df.fillna(method='ffill', inplace=True)

        # Normalize numerical columns to range [0, 1]
        scaler = MinMaxScaler()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # One-hot encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            encoder = OneHotEncoder()
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
            
            # Drop original categorical columns and merge with encoded ones
            df.drop(columns=categorical_cols, inplace=True)
            df = pd.concat([df, encoded_df], axis=1)
        
        return df

    def __len__(self):
        return max(0, len(self.dataframe) - self.seq_len + 1)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx:idx+self.seq_len]
        return torch.Tensor(x.values)  # return only x values