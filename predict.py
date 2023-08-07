import pandas as pd
import argparse
import numpy as np
from typing import Any
import torch
from model import LSTMModel
import pickle
from dataloader import TimeSeriesDataset_forPredict, SolarDataset
from torch.utils.data import DataLoader, Dataset
import glob
import os
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


LOAD_CKPT = "path to load_checkpoint"
LOAD_HP = "path to load_hyper-parameters"
LOAD_TEST = "path to testdata"
PV_CKPT = "./pv/checkpoints/*.pt"
PV_HP = "./pv/results/*.pkl"
PV_TEST = "./pv/eval_sequence/*.csv"
DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DAY ='0831'


parser = argparse.ArgumentParser(description="Prediction for PV and LOAD")
parser.add_argument('--mode', type=str, required=True,
                    help="Only 'pv' and 'load' are accepted. Should not be empty.")

args = parser.parse_args()


def load_checkpoint(ckpt_path: str, hp_path: str):
    with open(hp_path, 'rb') as f:
        results = pickle.load(f)
    
    hyperparameters = results['Hyperparameters']
    scalers = results['Scalers']    
        
    config = {
        'input_size': 17 if args.mode=='pv' else 63,
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
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict']) if args.mode=='pv' else model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    return model, scalers

def pv_predict(datapath: str, model, scaler):
    model.eval()
    
    df = pd.read_csv(datapath)
    dataset = SolarDataset(df)
    eval_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    
    for seq, _ in eval_loader:
        pred = model(seq)
        prediction_array = pred.detach().cpu().numpy()
        prediction = scaler.inverse_transform(prediction_array)
        rounded_data = np.maximum(prediction, 0.0).reshape(-1,1)
        
    return rounded_data.flatten()
    


def load_predict(datapath: str, model, scalers):
    model.eval()
    
    df = pd.read_excel(datapath)
    dataset = TimeSeriesDataset_forPredict(df)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    # Assuming df is the original dataset and it includes a 'date' column
    building_names = df.columns[-56:]  

    # Prepare storage for predictions
    predictions = []


    # Iterate over test set
    for sequence in test_loader:
        # Move sequence to correct device
        sequence = sequence.to(device)
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence).cpu().numpy()

        prediction_res = prediction.squeeze(0).reshape(24, 56)
        padding = np.zeros((prediction_res.shape[0], 7))
        prediction_pad = np.hstack((padding, prediction_res))
        prediction_inv = scalers.inverse_transform(prediction_pad)
        prediction_inv = np.delete(prediction_inv, np.s_[:7], axis=1)
        prediction = prediction_inv.reshape(prediction.shape)

        # Store the prediction
        predictions.append(prediction)

    # Combine all predictions
    predictions = np.concatenate(predictions, axis=0)

    # Create a DataFrame for predictions
    # Reshape the predictions to align with the number of building_names
    predictions = predictions.reshape(-1, len(building_names))
    predictions_df = pd.DataFrame(predictions, columns=building_names)

    predictions_df['total(KW)'] = predictions_df.sum(axis=1)

    # Save to Excel file
    output_filepath = './load/predict_for_'+TARGET_DAY+'.xlsx'  # adjust this as necessary
    predictions_df.to_excel(output_filepath, index=False)

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
        model, scaler = load_checkpoint(LOAD_CKPT, LOAD_HP)
        load_predict(LOAD_TEST, model, scaler)
    elif args.mode == 'pv':
        models = dict()
        scalers = dict()
        predictions = dict()
        
        for ckpt, hp, datapath in zip(sorted(glob.glob(PV_CKPT)), sorted(glob.glob(PV_HP)), sorted(glob.glob(PV_TEST))):
            b = ckpt.split("/")[-1].split("-")[0]
            models[b], scalers[b] = load_checkpoint(ckpt, hp)
            predictions[b] = pv_predict(datapath, models[b], scalers[b])
        
        output_filepath = os.path.join('.','summed-2022-08-31.csv')
        pd.DataFrame(predictions).to_csv(output_filepath, index=False)
        logger.info(f"Prediction completed. filepath: {output_filepath}")
    else:
        raise ValueError("--mode excepts either 'pv' or 'load'")         
