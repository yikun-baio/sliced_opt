import numpy as np
import torch


path = "./saved/"
out_path = "./saved_3/"
ext = ".pt"
data_names = ['dragon', 'mumble_sitting', 'stanford_bunny', 'witchcastle']

# for name in data_names :
#     data = torch.load(path + name + ext)
#     keys = []
#     for key in data : 
#         keys.append(key)
#     for key in keys :
#         if key[0] == "X":
#             new_key = key.replace("X", "Y")
#             data[new_key] = data.pop(key)
#         elif key[0] == "Y":
#             new_key = key.replace("Y", "X")
#             data[new_key] = data.pop(key)
#         else:
#             continue
        
#     torch.save(data, out_path + name + ext)
    
    
## Testing : 
for name in data_names :
    data = torch.load(path + name + ext)
    for key in data : 
        print(key)
    print()