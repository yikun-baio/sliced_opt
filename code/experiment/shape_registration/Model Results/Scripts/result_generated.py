from matplotlib.pyplot import sca
import helpers as hlp
import torch
import numpy as np

# For 10k-5p: X1-5p, Y10-5p
# For 10k-7p: X1-7p, Y10-7p
# For 9k-5p: X1-5p, Y11-5p
# For 9k-7p: X1-7p, Y11-7p
def get_type(typ):
    if typ == "9k-5p":
        return "X1-5p", "Y11-5p"
    if typ == "9k-7p":
        return "X1-7p", "Y11-7p"
    if typ == "10k-5p":
        return "X1-5p", "Y10-5p"
    if typ == "10k-7p":
        return "X1-7p", "Y10-7p"
    

def open_paras(path, model_name) : 
    param_list = torch.load(path + model_name + ".pt")
    return param_list

def generating_results(parameter, data_path, X1, Y1, saved_path) : 
    N = len(parameter)
    data = hlp.load_data(data_path)
    X, Y = hlp.process_data(data, X1, Y1)
    Y = torch.from_numpy(Y)
    hlp.generate(X, Y, saved_path, "0")
    
    for i in range(N):
        if i % 500 == 0 or i == N-1 :
            print(f'Epoch : {i}/{N}')
        if i <= 200 or i % 20 == 0 or i == N-1 :
            param = parameter[i]
            rotation = param['rotation']
            scalar = param['scalar']
            beta = param['beta']
            X1_hat = Y@rotation * scalar + beta
            hlp.generate(X, X1_hat, saved_path, name = str(i))
             
    


