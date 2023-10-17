'''
this script is used for evaluate Monte Carlo experiments
'''
#%%
import os
import numpy as np
from plotmain import *

def evaluateSWMarg(subfolders, sw_file_name):
    rms_pos, rms_att = [], []

    for subfolder in subfolders:
        result_file = datapath + subfolder + "/" + sw_file_name
        results = np.loadtxt(result_file)

        rms_list_pos, rms_list_att = [], []
        # pos = results[:, 1]
        # print(RMS(np.zeros(pos.shape), pos))
        # print(np.linalg.norm(pos, 2, 1).shape)
        # RMS at each epoch
        for i in range(results.shape[0]):
            pos_data, att_data =  results[0: i + 1, 1: 4], results[0: i + 1, 4: 7]
            # pos_data, att_data =  np.linalg.norm(results[0: i + 1, 1: 4], 2, 1), np.linalg.norm(results[0: i + 1, 4: 7], 2, 1)
            rms_list_pos.append(RMS(np.zeros(pos_data.shape), pos_data))
            rms_list_att.append(RMS(np.zeros(att_data.shape), att_data))
            # rms_list_pos.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 1: 4].reshape((i + 1) * 3, -1)))
            # rms_list_att.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 4: 7].reshape((i + 1) * 3, -1)))
        
        rms_pos.append(np.array(rms_list_pos))
        rms_att.append(np.array(rms_list_att))
    # print (rms_pos)
    mean_rms_pos = np.mean(np.array(rms_pos).transpose(), 1)
    mean_rms_att = np.mean(np.array(rms_att).transpose(), 1)
    # print(mean_rms_pos)
    return results[:, 0], mean_rms_pos, mean_rms_att

#%% initialize parameters
basepath = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/data_2/"
datapath = basepath + "data/"
subfolders = sorted(os.listdir(datapath), key=lambda x : int(x))
savepath = basepath + "results/" 
#%% SWO and MSWF experiment
SWO_file_name = "result.txt.-1s.CLS_SW_Marg1.Noise"
time, mean_rms_pos_swo, mean_rms_att_swo = evaluateSWMarg(subfolders, SWO_file_name)
#%%
MSWF_file_name = "result.txt.-1s.Filter_SW_Marg1.Noise"
time, mean_rms_pos_mswf, mean_rms_att_mswf = evaluateSWMarg(subfolders, MSWF_file_name)

# define attribute
logyAttributes = {}
logyAttributes["ylabel"] = "Relative Error [deg]"
logyAttributes["xlabel"] = "time [s]"
logyAttributes["legend"] = ["Attitude"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["ylim"] = [1E-12, 1E-6]
plotLogY(time - time[0], [np.abs(mean_rms_att_mswf - mean_rms_att_swo)], savepath + "MSWF-SWO-Att.svg", logyAttributes, False)

logyAttributes = {}
logyAttributes["ylabel"] = "Relative Error [m]"
logyAttributes["xlabel"] = "time [s]"
logyAttributes["legend"] = ["Position"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["ylim"] = [1E-13, 1E-7]
plotLogY(time - time[0], [np.abs(mean_rms_pos_mswf - mean_rms_pos_swo)], savepath + "MSWF-SWO-pos.svg", logyAttributes, False)

#%% SWO and normal SWF
SWF_file_name = "result.txt.-1s.Filter_SW.Noise"
time, mean_rms_pos_swf, mean_rms_att_swf = evaluateSWMarg(subfolders, SWF_file_name)

posAttributes = {}
posAttributes["ylabel"] = "Position RMSE [m]"
posAttributes["xlabel"] = "time [s]"
posAttributes["legend"] = ["SWO", "SWF"]
posAttributes["xlim"] = [0, 60]
posAttributes["ylim"] = [-1, 1]
posAttributes["scientific"] = True
swf_swo_compare_pos = np.append(mean_rms_pos_swo.reshape(-1, 1), mean_rms_pos_swf.reshape(-1, 1), axis=1)
ploterror(time - time[0], swf_swo_compare_pos, savepath + "RMSE-SWO-SWF-pos.svg", posAttributes, False)

attAttributes = {}
attAttributes["ylabel"] = "Attitude RMSE [Deg]"
attAttributes["xlabel"] = "time [s]"
attAttributes["legend"] = ["SWO", "SWF"]
attAttributes["xlim"] = [0, 60]
attAttributes["ylim"] = [-4, 4]
attAttributes["scientific"] = True
swf_swo_compare_att = np.append(mean_rms_att_swo.reshape(-1, 1), mean_rms_att_swf.reshape(-1, 1), axis=1)
ploterror(time - time[0], swf_swo_compare_att, savepath + "RMSE-SWO-SWF-att.svg", attAttributes, False)

# %% LSE vs. full-state EKF
LSE_file_name = "result.txt.3s.CLS_Seq.Noise"
time, mean_rms_pos_lse, mean_rms_att_lse = evaluateSWMarg(subfolders, LSE_file_name)
print("----------------------------\n\n\n\n\n\n\n\n")
EKF_file_name = "result.txt.3s.FilterAllState.Noise"
time, mean_rms_pos_ekf, mean_rms_att_ekf = evaluateSWMarg(subfolders, EKF_file_name)

print(mean_rms_pos_lse-mean_rms_pos_ekf)

# print(np.mean(np.array(rms_pos_test)) - np.mean(np.array(rms_pos_test1)))


# diff = (np.array(rms_pos_test) - np.array(rms_pos_test1)).transpose()

# print(np.max(diff))
# print(np.unravel_index(np.argmax(diff), diff.shape))
# # np.savetxt("./log/tms_test.txt", np.array(rms_pos_test) - np.array(rms_pos_test1))
# np.savetxt("./log/tms_test1", rms_pos_test1)
logyAttributes = {}
logyAttributes["ylabel"] = "Relative Error [deg]"
logyAttributes["xlabel"] = "time [s]"
logyAttributes["legend"] = ["Attitude"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["ylim"] = [1E-8, 1E-5]
plotLogY(time - time[0], [np.abs(mean_rms_att_lse - mean_rms_att_ekf)], savepath + "fullstate-Att.svg", logyAttributes, False)

logyAttributes = {}
logyAttributes["ylabel"] = "Relative Error [m]"
logyAttributes["xlabel"] = "time [s]"
logyAttributes["legend"] = ["Position"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["ylim"] = [1E-9, 1E-5]
plotLogY(time - time[0], [np.abs(mean_rms_pos_lse - mean_rms_pos_ekf)], savepath + "fullstate-pos.svg", logyAttributes, False)


# %%
