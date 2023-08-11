import pandas as pd
import argparse
import numpy as np
from typing import Any
import torch
from model import LSTMModel
import pickle
from dataloader import TimeSeriesDataset, SolarDataset
from torch.utils.data import DataLoader, Dataset
import glob
import os
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


LOAD_CKPT = "./load/best_results/model_20230807_165319.pt"
LOAD_HP = "./load/best_results/20230807_165319.pkl"
LOAD_TEST = "./load/data/merged_data_KW.xlsx"
PV_CKPT = "./pv/checkpoints/*.pt"
PV_HP = "./pv/results/*.pkl"
PV_TEST = "./pv/eval_sequence/*.csv"
DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DAY ='0831'


parser = argparse.ArgumentParser(description="Prediction for PV and LOAD")
parser.add_argument('--mode', type=str, required=True,
                    help="Only 'pv' and 'load' are accepted. Should not be empty.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(ckpt_path: str, hp_path: str):
    with open(hp_path, 'rb') as f:
        results = pickle.load(f)
    
    hyperparameters = results['Hyperparameters']  
        
    config = {
        'input_size': 17 if args.mode=='pv' else 7,
        'hidden_size': hyperparameters['hidden_size'] if args.mode=='pv' else hyperparameters['hidden_dim'],
        'num_layers': hyperparameters['num_layers'] if args.mode=='pv' else hyperparameters['n_layers'],
        'output_size': 24 if args.mode=='pv' else 24*56
    }
    
    model = LSTMModel(
        config['input_size'], 
        config['hidden_size'], 
        config['num_layers'], 
        config['output_size']
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if args.mode=='pv' else checkpoint) # else model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    return model

def pv_predict(datapath: str, model):
    model.eval()
    
    df = pd.read_csv(datapath)
    dataset = SolarDataset(df)
    eval_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    
    for seq, label in eval_loader:
        pred = model(seq.to(device))
        prediction_array = pred.detach().cpu().numpy()
        target_array = label.detach().cpu().numpy()
        prediction = dataset.y_scaler.inverse_transform(prediction_array)
        target = dataset.y_scaler.inverse_transform(target_array)
        rounded_data = np.maximum(prediction, 0.0).reshape(-1,1)
        
    return rounded_data.flatten(), target.flatten()
    


def load_predict(datapath: str, model):
    
    df = pd.read_excel(datapath)
    dataset = TimeSeriesDataset(df)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    real_sequence, real_target = next(iter(test_loader))
    
    # Assuming df is the original dataset and it includes a 'date' column
    building_names = df.columns[-56:]  

    
    # 모델을 평가 모드로 전환합니다.
    model.eval()

    # 예측을 수행합니다.
    with torch.no_grad():
        prediction = model(real_sequence)

    print(prediction.shape, real_target.shape)
    prediction = prediction.squeeze(0).reshape(24, 56).numpy()
    real_target = real_target.view(24, 56).numpy()

    # 패딩을 추가합니다.
    padding = np.zeros((prediction.shape[0], 7))
    prediction_pad = np.hstack((padding, prediction))
    real_target_pad = np.hstack((padding, real_target))
    # print(prediction_pad.shape, real_target_pad.shape)

    # 역변환을 적용하여 정규화를 해제합니다.
    prediction_inv = dataset.scaler.inverse_transform(prediction_pad)
    real_target_inv = dataset.scaler.inverse_transform(real_target_pad)

    # 처음 7개의 컬럼을 삭제합니다.
    prediction_inv = np.delete(prediction_inv, np.s_[:7], axis=1)
    real_target_inv = np.delete(real_target_inv, np.s_[:7], axis=1)

    # 원래의 형태로 다시 변형합니다.
    prediction = prediction_inv.reshape(prediction.shape)
    real_target = real_target_inv.reshape(real_target.shape)

    # 에러(실제 목표값과 예측값의 차이)를 계산합니다.
    error = real_target - prediction

    # 예측값을 위한 DataFrame을 생성합니다.
    predicted_df = pd.DataFrame(prediction, columns=building_names)
    real_target_df = pd.DataFrame(real_target, columns=building_names)
    error_df = pd.DataFrame(error, columns=building_names)
    err_total = error_df.values.flatten().sum()

    # 성능 지표를 계산합니다.
    mae_n = mean_absolute_error(real_target, prediction)
    mse_n = mean_squared_error(real_target, prediction)
    rmse_n = math.sqrt(mse_n)
    logger.info(f'MAE: {mae_n:.4f}, MSE: {mse_n:.4f}, RMSE: {rmse_n:.4f} (no normalization)')
    logger.info(f"err_total: {err_total}")

    
    # Save to Excel file
    output_filepath = f'./{args.mode}_predict_for_'+TARGET_DAY+'.xlsx'  # adjust this as necessary
    predicted_df.to_excel(output_filepath, index=False)

    logger.info(f"Prediction completed. filepath: {output_filepath}")
    
    

if __name__=='__main__':
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    pass
    "***TO DO***"
    """
    PV_pred: load_checkpoint, pv_predict INSIDE the loop
    LOAD_pred: load_checkpoint, load_predict at once
    
    """
    logger.info("Predicting {} for {}...".format(args.mode.upper(), TARGET_DAY))
    if args.mode == 'load':
        model = load_checkpoint(LOAD_CKPT, LOAD_HP)
        load_predict(LOAD_TEST, model)
    elif args.mode == 'pv':
        models = dict()
        predictions = dict()
        target = dict()
        
        for ckpt, hp, datapath in zip(sorted(glob.glob(PV_CKPT)), sorted(glob.glob(PV_HP)), sorted(glob.glob(PV_TEST))):
            b = ckpt.split("/")[-1].split("-")[0]
            models[b] = load_checkpoint(ckpt, hp)
            predictions[b], target[b] = pv_predict(datapath, models[b])
        
        error = pd.DataFrame(target)-pd.DataFrame(predictions)
        logger.info(f"Total Prediction Error: {error.sum().sum()}")
        
        output_filepath = os.path.join('.',f'{args.mode}_predict_for_'+TARGET_DAY+'.csv')
        pd.DataFrame(predictions).to_csv(output_filepath, index=False)
        logger.info(f"Prediction completed. filepath: {output_filepath}")
    else:
        raise ValueError("--mode excepts either 'pv' or 'load'")         
