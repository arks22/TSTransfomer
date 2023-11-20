import numpy as np
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset
from process_dummy import generate_dummy_data


def get_dataloader(params):
    batch_size = params['batch_size']
    
    if params['data_name'] == 'dummy':
        train_timeseries, train_labels, val_timeseries, val_labels, test_timeseries, test_labels = generate_dummy_data(100000)  
    elif params['data_name'] == 'btc':
        train_timeseries = np.load(params['train_timeseries_path'])
        train_labels     = np.load(params['train_labels_path'])
        val_timeseries   = np.load(params['val_timeseries_path'])
        val_labels       = np.load(params['val_labels_path'])
        test_timeseries  = np.load(params['test_timeseries_path'])
        test_labels      = np.load(params['test_labels_path'])
        
        finetune_timeseries = np.load(params['finetune_timeseries_path'])
        finetune_labels     = np.load(params['finetune_labels_path'])
        
    if params['finetune']:
        finetune_dataset = TimeSeriesDataset(train_timeseries, train_labels)
        finetune_dataloader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    else:
        finetune_dataloader = None

    train_dataset = TimeSeriesDataset(finetune_timeseries, finetune_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TimeSeriesDataset(val_timeseries, val_labels) 
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(test_timeseries, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train Data: {len(train_dataset)} | Finetune Data:{len(finetune_dataloader)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    return train_dataloader, finetune_dataloader, val_dataloader, test_dataloader