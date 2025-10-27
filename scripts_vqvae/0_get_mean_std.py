import sys
sys.path.append('./')
import numpy as np
import lmdb as lmdb
import pandas as pd
import torch
from termcolor import colored
from loguru import logger



from utils import rotation_conversions as rc



def main():
    #training_speakers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    training_speakers = [2]
    split_rule = pd.read_csv("./dataset/BEAT2/beat_english_v2.0.0/train_test_split.csv")
    selected_file_train = split_rule.loc[(split_rule['type'] == "train") & (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))]
    selected_file_test= split_rule.loc[(split_rule['type'] == "test") & (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))]
    selected_file_val = split_rule.loc[(split_rule['type'] == "val") & (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))]
    selected_file = pd.concat([selected_file_train, selected_file_test,selected_file_val])

    all_trans_data = []
    all_pose_data = []
    for index, file_name in selected_file.iterrows():
        f_name = file_name["id"]
        ext = ".npz"
        pose_file = "./dataset/BEAT2/beat_english_v2.0.0/smplxflame_30" + "/" + f_name + ext
        pose_each_file = []
        trans_each_file = []
        id_pose = f_name #1_wayne_0_1_1
        
        logger.info(colored(f"# ---- Load Pose   {id_pose} ---- #", "blue"))

        pose_data = np.load(pose_file, allow_pickle=True)

        pose_each_file = torch.from_numpy(pose_data["poses"])
        pose_each_file = rc.axis_angle_to_matrix(pose_each_file.reshape(-1,55,3))
        pose_each_file = rc.matrix_to_rotation_6d(pose_each_file)
        pose_each_file = pose_each_file.numpy()
        all_pose_data.append(pose_each_file.reshape(-1, 55*6))


        trans_each_file = pose_data["trans"]
        n,_ = trans_each_file.shape
        trans_x = trans_each_file[:,0]
        trans_y = trans_each_file[:,1]
        trans_z = trans_each_file[:,2]
        trans_x = trans_x - trans_x[0]
        trans_z = trans_z - trans_z[0]
        
        v_x = np.zeros(n)
        v_z = np.zeros(n) 
        v_x_tmp = trans_x[1:] - trans_x[:-1]
        v_z_tmp = trans_z[1:] - trans_z[:-1]
        v_x[1:] = v_x_tmp
        v_z[1:] = v_z_tmp
        v_x[0]=v_x[1]
        v_z[0]=v_z[1]
        
        v_x=v_x.reshape(-1,1)
        trans_y=trans_y.reshape(-1,1)
        v_z=v_z.reshape(-1,1)
        
        trans_each_file = np.concatenate([v_x,trans_y,v_z],axis=1)
        all_trans_data.append(trans_each_file.reshape(-1, 3))
        



    all_pose_data = np.concatenate(all_pose_data, axis=0)
    beatx_pose_mean = np.mean(all_pose_data, axis=0)
    beatx_pose_std = np.std(all_pose_data, axis=0)

    np.save('mean_std/beat2_2_330_mean.npy', beatx_pose_mean)
    np.save('mean_std/beat2_2_330_std.npy', beatx_pose_std)



    all_trans_data = np.concatenate(all_trans_data, axis=0)
    beatx_trans_mean = np.mean(all_trans_data, axis=0)
    beatx_trans_std = np.std(all_trans_data, axis=0)
    np.save('mean_std/beat2_2_trans_mean.npy', beatx_trans_mean)
    np.save('mean_std/beat2_2_trans_std.npy', beatx_trans_std)


if __name__ == "__main__":
    main()
    

