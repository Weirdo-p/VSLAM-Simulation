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
# MSWF_file_name = "result.txt.-1s.Filter_SW_Marg1.Noise"
# time, mean_rms_pos_mswf, mean_rms_att_mswf = evaluateSWMarg(subfolders, MSWF_file_name)

# # define attribute
# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (deg)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-12, 1E-6]
# plotLogY(time - time[0], [np.abs(mean_rms_att_mswf - mean_rms_att_swo)], savepath + "MSWF-SWO-Att.svg", logyAttributes, False)

# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (m)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Position"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-13, 1E-7]
# plotLogY(time - time[0], [np.abs(mean_rms_pos_mswf - mean_rms_pos_swo)], savepath + "MSWF-SWO-pos.svg", logyAttributes, False)

# #%% SWO and normal SWF
# SWF_file_name = "result.txt.-1s.Filter_SW.Noise"
# time, mean_rms_pos_swf, mean_rms_att_swf = evaluateSWMarg(subfolders, SWF_file_name)

# posAttributes = {}
# posAttributes["ylabel"] = "Position RMSE (m)"
# posAttributes["xlabel"] = "time (s)"
# posAttributes["legend"] = ["SWO", "SWF"]
# posAttributes["xlim"] = [0, 60]
# posAttributes["ylim"] = [-1, 1]
# posAttributes["scientific"] = True
# swf_swo_compare_pos = np.append(mean_rms_pos_swo.reshape(-1, 1), mean_rms_pos_swf.reshape(-1, 1), axis=1)
# ploterror(time - time[0], swf_swo_compare_pos, savepath + "RMSE-SWO-SWF-pos.svg", posAttributes, False)

# attAttributes = {}
# attAttributes["ylabel"] = "Attitude RMSE (Deg)"
# attAttributes["xlabel"] = "time (s)"
# attAttributes["legend"] = ["SWO", "SWF"]
# attAttributes["xlim"] = [0, 60]
# attAttributes["ylim"] = [-4, 4]
# attAttributes["scientific"] = True
# swf_swo_compare_att = np.append(mean_rms_att_swo.reshape(-1, 1), mean_rms_att_swf.reshape(-1, 1), axis=1)
# ploterror(time - time[0], swf_swo_compare_att, savepath + "RMSE-SWO-SWF-att.svg", attAttributes, False)

# # %% LSE vs. full-state EKF
# LSE_file_name = "result.txt.3s.CLS_Seq.Noise"
# time, mean_rms_pos_lse, mean_rms_att_lse = evaluateSWMarg(subfolders, LSE_file_name)
# EKF_file_name = "result.txt.3s.FilterAllState.Noise"
# time, mean_rms_pos_ekf, mean_rms_att_ekf = evaluateSWMarg(subfolders, EKF_file_name)

# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (deg)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-8, 1E-5]
# plotLogY(time - time[0], [np.abs(mean_rms_att_lse - mean_rms_att_ekf)], savepath + "fullstate-Att.svg", logyAttributes, False)

# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (m)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Position"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-9, 1E-5]
# plotLogY(time - time[0], [np.abs(mean_rms_pos_lse - mean_rms_pos_ekf)], savepath + "fullstate-pos.svg", logyAttributes, False)


# %% new plots
def evaluateRMS(subfolders, sw_file_name):
    rms_pos, rms_att = [], []

    for subfolder in subfolders:
        result_file = datapath + subfolder + "/" + sw_file_name
        results = np.loadtxt(result_file)

        rms_list_pos, rms_list_att = [], []
        # pos = results[:, 1]
        # print(RMS(np.zeros(pos.shape), pos))
        # print(np.linalg.norm(pos, 2, 1).shape)
        # RMS at each epoch
        # for i in range(results.shape[0]):
        pos_data, att_data =  results[:, 1: 4], results[:, 4: 7]
        # pos_data, att_data =  np.linalg.norm(results[0: i + 1, 1: 4], 2, 1), np.linalg.norm(results[0: i + 1, 4: 7], 2, 1)
        rms_pos.append(RMS(np.zeros(pos_data.shape), pos_data))
        rms_att.append(RMS(np.zeros(att_data.shape), att_data))
        # rms_list_pos.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 1: 4].reshape((i + 1) * 3, -1)))
        # rms_list_att.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 4: 7].reshape((i + 1) * 3, -1)))
        
    #     rms_pos.append(np.array(rms_list_pos))
    #     rms_att.append(np.array(rms_list_att))
    # # print (rms_pos)
    # mean_rms_pos = np.mean(np.array(rms_pos).transpose(), 1)
    # mean_rms_att = np.mean(np.array(rms_att).transpose(), 1)
    # print(mean_rms_pos)
    return results[:, 0], rms_pos, rms_att

#%% initialize parameters
basepath = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/data_2/"
datapath = basepath + "data/"
subfolders = sorted(os.listdir(datapath), key=lambda x : int(x))
savepath = basepath + "results/" 

#%% SWO and MSWF experiment
SWO_file_name = "result.txt.-1s.CLS_SW_Marg1.Noise"
MSWF_file_name = "result.txt.-1s.Filter_SW_Marg1.Noise"

time, mean_rms_pos_swo, mean_rms_att_swo = evaluateRMS(subfolders, SWO_file_name)
time, mean_rms_pos_mswf, mean_rms_att_mswf = evaluateRMS(subfolders, MSWF_file_name)
SWO_MSWF_POS = np.array([mean_rms_pos_swo, mean_rms_pos_mswf]).transpose()
SWO_MSWF_ATT = np.array([mean_rms_att_swo, mean_rms_att_mswf]).transpose()

#%% SWO and normal SWF
SWF_file_name = "result.txt.-1s.Filter_SW.Noise"
time, mean_rms_pos_swf, mean_rms_att_swf = evaluateRMS(subfolders, SWF_file_name)
SWO_SWF_POS = np.array([mean_rms_pos_swo, mean_rms_pos_swf]).transpose()
SWO_SWF_ATT = np.array([mean_rms_att_swo, mean_rms_att_swf]).transpose()

# %% LSE vs. full-state EKF
LSE_file_name = "result.txt.10s.CLS_Seq.Noise"
time, mean_rms_pos_lse, mean_rms_att_lse = evaluateRMS(subfolders, LSE_file_name)
EKF_file_name = "result.txt.10s.FilterAllState.Noise"
time, mean_rms_pos_ekf, mean_rms_att_ekf = evaluateRMS(subfolders, EKF_file_name)
BATCH_SEQ_POS = np.array([mean_rms_pos_lse, mean_rms_pos_ekf]).transpose()
BATCH_SEQ_ATT = np.array([mean_rms_att_lse, mean_rms_att_ekf]).transpose()

#%% plot
dot_dict = {}
print(SWO_MSWF_POS.shape)
# print(np.array([mean_rms_pos_swo, mean_rms_pos_mswf]))
dot_dict["SWO vs. SWF-SA"] = SWO_MSWF_POS
dot_dict["Batch vs. Seq"] = BATCH_SEQ_POS
dot_dict["SWO vs. SWF"] = SWO_SWF_POS
logyAttributes = {}
logyAttributes["ylabel"] = "RMS Of Filtering (m)"
logyAttributes["xlabel"] = "RMS Of Optimization (m)"
logyAttributes["legend"] = ["Attitude"]
logyAttributes["xlim"] = [0, 0.6]
logyAttributes["ylim"] = [0, 0.6]

# plotCompareDot(dot_dict, logyAttributes, "{0}results/total_pos.svg".format(basepath))

dot_dict = {}
print(SWO_MSWF_POS.shape)
# print(np.array([mean_rms_pos_swo, mean_rms_pos_mswf]))
dot_dict["SWO vs. SWF-SA"] = SWO_MSWF_ATT
dot_dict["Batch vs. Seq"] = BATCH_SEQ_ATT
dot_dict["SWO vs. SWF"] = SWO_SWF_ATT
logyAttributes = {}
logyAttributes["ylabel"] = "RMS Of Filtering (deg)"
logyAttributes["xlabel"] = "RMS Of Optimization (deg)"
logyAttributes["legend"] = ["Attitude"]
logyAttributes["xlim"] = [0, 1.5]
logyAttributes["ylim"] = [0, 1.5]

plotCompareDot(dot_dict, logyAttributes, "{0}results/total_att.svg".format(basepath))

#%%

def evaluateAveDiscrepancy(subfolders, swo_file_name, swf_file_name):
    ave_pos, ave_att, time = [], [], 0

    for subfolder in subfolders:
        swo_file = datapath + subfolder + "/" + swo_file_name
        swf_file = datapath + subfolder + "/" + swf_file_name
        swo = np.loadtxt(swo_file)
        swf = np.loadtxt(swf_file)
        time = swf[:, 0]
        swo_pos, swo_att =  swo[:, 1: 4], swo[:, 4: 7]
        swf_pos, swf_att =  swf[:, 1: 4], swf[:, 4: 7]
        test = swo_pos - swf_pos
        if np.any(np.abs(test) > 1E-4):
            print("test")
        if np.any(np.linalg.norm(swo_pos - swf_pos, 2, axis=1) > 1E-4):
            print(test)
            # print()
            print("test")
        ave_pos.append(np.linalg.norm(swo_pos - swf_pos, 2, axis=1))
        ave_att.append(np.linalg.norm(swo_att - swf_att, 2, axis=1))

    ave_pos = np.mean(ave_pos, axis=0)
    ave_att = np.mean(ave_att, axis=0)
    print("test")
    return time, ave_pos, ave_att


#%%
MSWF_file_name = "result.txt.-1s.Filter_SW_Marg1.Noise"
SWO_file_name = "result.txt.-1s.CLS_SW_Marg1.Noise"

time, mean_pos_mswf_swo, mean_att_mswf_swo = evaluateAveDiscrepancy(subfolders, SWO_file_name, MSWF_file_name)

# define attribute
logyAttributes = {}
logyAttributes["y1label"] = "Position (m)"
logyAttributes["y2label"] = "Attitude (deg)"
logyAttributes["xlabel"] = "time (s)"
logyAttributes["legend"] = ["Position", "Attitude"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["y2lim"] = [1E-12, 1E-6]
logyAttributes["y1lim"] = [1E-13, 1E-7]
# plotLogY_Twiny(time - time[0], [mean_pos_mswf_swo], [mean_att_mswf_swo], savepath + "MSWF-SWO.svg", logyAttributes, False)


LSE_file_name = "result.txt.10s.CLS_Seq.Noise"
EKF_file_name = "result.txt.10s.FilterAllState.Noise"
time, mean_pos_lse_ekf, mean_att_lse_ekf = evaluateAveDiscrepancy(subfolders, LSE_file_name, EKF_file_name)

logyAttributes = {}
logyAttributes["y1label"] = "Position (m)"
logyAttributes["y2label"] = "Attitude (deg)"
logyAttributes["xlabel"] = "time (s)"
logyAttributes["legend"] = ["Position", "Attitude"]
logyAttributes["xlim"] = [0, 60]
logyAttributes["y2lim"] = [1E-8, 1E-3]
logyAttributes["y1lim"] = [1E-8, 1E-3]
plotLogY_Twiny(time - time[0], [mean_pos_lse_ekf], [mean_att_lse_ekf], savepath + "fullstate.svg", logyAttributes, False)


# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (m)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Position"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-13, 1E-7]
# plotLogY(time - time[0], [mean_pos_mswf_swo], savepath + "MSWF-SWO-pos.svg", logyAttributes, False)

# #%% SWO and normal SWF
# SWF_file_name = "result.txt.-1s.Filter_SW.Noise"
# time, mean_rms_pos_swf, mean_rms_att_swf = evaluateSWMarg(subfolders, SWF_file_name)

# posAttributes = {}
# posAttributes["ylabel"] = "Position RMSE (m)"
# posAttributes["xlabel"] = "time (s)"
# posAttributes["legend"] = ["SWO", "SWF"]
# posAttributes["xlim"] = [0, 60]
# posAttributes["ylim"] = [-1, 1]
# posAttributes["scientific"] = True
# swf_swo_compare_pos = np.append(mean_rms_pos_swo.reshape(-1, 1), mean_rms_pos_swf.reshape(-1, 1), axis=1)
# ploterror(time - time[0], swf_swo_compare_pos, savepath + "RMSE-SWO-SWF-pos.svg", posAttributes, False)

# attAttributes = {}
# attAttributes["ylabel"] = "Attitude RMSE (Deg)"
# attAttributes["xlabel"] = "time (s)"
# attAttributes["legend"] = ["SWO", "SWF"]
# attAttributes["xlim"] = [0, 60]
# attAttributes["ylim"] = [-4, 4]
# attAttributes["scientific"] = True
# swf_swo_compare_att = np.append(mean_rms_att_swo.reshape(-1, 1), mean_rms_att_swf.reshape(-1, 1), axis=1)
# ploterror(time - time[0], swf_swo_compare_att, savepath + "RMSE-SWO-SWF-att.svg", attAttributes, False)

# %% LSE vs. full-state EKF
# LSE_file_name = "result.txt.3s.CLS_Seq.Noise"
# time, mean_rms_pos_lse, mean_rms_att_lse = evaluateSWMarg(subfolders, LSE_file_name)
# EKF_file_name = "result.txt.3s.FilterAllState.Noise"
# time, mean_rms_pos_ekf, mean_rms_att_ekf = evaluateSWMarg(subfolders, EKF_file_name)

# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (deg)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-8, 1E-5]
# plotLogY(time - time[0], [np.abs(mean_rms_att_lse - mean_rms_att_ekf)], savepath + "fullstate-Att.svg", logyAttributes, False)

# logyAttributes = {}
# logyAttributes["ylabel"] = "Discrepancy (m)"
# logyAttributes["xlabel"] = "time (s)"
# logyAttributes["legend"] = ["Position"]
# logyAttributes["xlim"] = [0, 60]
# logyAttributes["ylim"] = [1E-9, 1E-5]
# plotLogY(time - time[0], [np.abs(mean_rms_pos_lse - mean_rms_pos_ekf)], savepath + "fullstate-pos.svg", logyAttributes, False)
