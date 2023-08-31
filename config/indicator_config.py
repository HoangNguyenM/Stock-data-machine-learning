import os
import pickle
from skopt.space import Real, Integer, Categorical

def get_learnables(strategy_name):
    indicators = ['EMAshort', 'EMAlong', 'EMAperiod',  'MACDperiod', 
                'RSIoverSold', 'VWAPratio', 
                'Donperiod', 'consecutiveBreak','Conditionsnum', 
                'profit_ratio', 'loss_ratio']
    if strategy_name == 'CNN':
        indicators += ['priorDuration', 'postDuration', 'CNNHigh']

    return indicators

def get_space_config():

    space = {
        'EMAshort': Integer(7, 29, name='EMAshort'),
        'EMAlong': Integer(30, 80, name='EMAlong'),
        'Donperiod': Integer(15, 120, name='Donperiod'),
        'RSIoverSold': Real(15, 95, name='RSIoverSold'), 
        'VWAPratio': Real(0.4, 2.0, name='VWAPratio'), 
        'EMAperiod': Integer(0, 10, name='EMAperiod'),
        'MACDperiod': Integer(0, 10, name='MACDperiod'),
        'consecutiveBreak': Integer(0, 3, name='consecutiveBreak'),
        'Conditionsnum': Integer(1, 3, name='Conditionsnum'),

        'profit_ratio': Real(2.0, 10.0, name='profit_ratio'), 
        'loss_ratio': Real(0.5, 4.0, name='loss_ratio'),

        'priorDuration': Categorical([30, 45, 60, 90, 120], name='priorDuration'),
        'postDuration': Categorical([15, 30, 45], name='postDuration'),
        'CNNHigh': Real(0.00, 0.50, name='CNNHigh')
    }

    return space

def get_best_config():

    best = {
        'EMAshort': 10,
        'EMAlong': 40,
        'Donperiod': 20,
        'RSIoverSold': 48, 
        'VWAPratio': 0.83, 
        'EMAperiod': 9,
        'MACDperiod': 5,
        'consecutiveBreak': 1,
        'Conditionsnum': 2,
        'profit_ratio': 5, 
        'loss_ratio': 2
    }

    # Get a list of all .pkl files in the folder
    folder_path = "config"
    pkl_files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]
    
    # Initialize an empty list to store loaded variables
    loaded_variables = [best]

    # Loop through the .pkl files
    for pkl_file in pkl_files:
        pkl_file_path = os.path.join(folder_path, pkl_file)
        with open(pkl_file_path, "rb") as file:
            loaded_data = pickle.load(file)
        
        loaded_variables.append(loaded_data)

    return loaded_variables
