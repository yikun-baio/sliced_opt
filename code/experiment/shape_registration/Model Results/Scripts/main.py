from result_generated import *
import argparse
import os

def main(args):
    param_list = open_paras(args['para_path'], args['model_name'])
    X, Y = get_type(args['data_version'])
    generating_results(param_list, data_path = args['data_path'] + args['data'], X1 = X, Y1 = Y, saved_path = args['saved_path'])
    
    
    
args = {}
    
models = ['sopt_param', 'spot_param', 'icp_du_param', 'icp_umeyama_param']
ver = ['9k-5p', '9k-7p', '10k-5p', '10k-7p']
datas = ['mumble_sitting']

for data in datas : 
    for model in models : 
        for version in ver : 
            print (f"************  Data : {data}\t Model : {model}\t Version : {version}")
            args['para_path'] = "../Parameters/" + data + "/" + version + "/"
            args['model_name'] = model
            args['data_path'] = "../Data/"
            args['data'] = data
            args['data_version'] = version
            args['saved_path'] = "../Images/"+ args['data'] + "/"  + args['data_version'] + "/" + args['model_name'] + "/"
            if not os.path.exists(args['saved_path']):
                os.makedirs(args['saved_path'])
            main(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--para_path', type=str,      default="../Parameters/mumble_sitting/9k-5p/", help = "Path of the Parameters")
#     parser.add_argument('--model_name', type = str, default = 'sopt_param', help = "Name of the model")
#     parser.add_argument('--data_path', type = str, default = "../Data/", help = "Path to Data")
#     parser.add_argument('--data', type = str, default = "mumble_sitting", help = "Name of the point cloud")
#     parser.add_argument('--saved_path', type = str, default = "../Images/", help = "Saved Path")
#     parser.add_argument('--data_version', type = str, default = "9k-5p", help = "Type of point cloud")
    
#     config = parser.parse_args()
#     args = vars(config)

#     args['saved_path'] = args['saved_path'] + args['data'] + "/"  + args['data_version'] + "/" + args['model_name'] + "/"
    
#     if not os.path.exists(args['saved_path']):
#         os.makedirs(args['saved_path'])
        
#     print('------------ Options -------------')
#     for key, value in sorted(args.items()):
#         print('%16.16s: %16.16s' % (str(key), str(value)))
#     print('-------------- End ----------------')

#     main(args)
