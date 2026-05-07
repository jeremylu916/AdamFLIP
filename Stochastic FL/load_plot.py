import numpy as np
from SOFL import *

data = np.load('logistic_regression.npz', allow_pickle=True)
print('NPZ keys:', data.files)

# Support either a single 'history' array or separate arrays
if 'history' in data.files:
    history = data['history']
else:
    f_values = data.get('f_values')
    KKT_gaps = data.get('KKT_gaps')
    h_violations = data.get('h_violations')
    history = list(zip(f_values, KKT_gaps, h_violations))

print('Loaded history length:', len(history))

plot_optimization_history_2(history, legends=['SOFL'], linewidth=3, savename='logistic_regression.png')