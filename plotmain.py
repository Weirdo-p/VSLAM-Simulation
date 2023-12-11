import glob
import os
from copyreg import add_extension
from cProfile import label
from matplotlib.collections import LineCollection
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def RMS(gt, obs):
    rms = np.sqrt(((gt - obs) ** 2).sum() / gt.shape[0])
    return rms

font={
    #   'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold'
}
font1={
    #   'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold'
}

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(255 / 255), (204 / 255), (102 / 255)],  # blue
         0: [(20 / 255), (169 / 255), (89 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
marker = ['o', 's', '^']

direc=["E", "N", "U"]
## for tj format

def ploterror_XYZ(time, neu, save, attribute):
    if neu.shape[0] < 1:
        return
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']
    # print(save)

    cm = plt.cm.get_cmap('ocean')
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        # ax = plt.subplot(3, 1, i + 1)
        # plt.tight_layout()
        # if i != 2:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        plt.grid(b=False, linestyle='--', color='k', alpha=0.4)
        direc_ = neu[:, i]
        sc = plt.scatter(time, (neu[:, i]), c=neu[:, i], ls="-", label=direc[i], linewidth=2, cmap=cm, vmin=0, vmax=11)#, marker=marker[i], markersize=4)

        print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.colorbar(sc)
    plt.ylabel(attribute['ylabel'], labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    isSubplot = attribute["subplot"]["bplot"]

    if isSubplot:
        subPlotAtt = attribute["subplot"]
        xpos, ypos, width, height = subPlotAtt["xpos"], subPlotAtt["ypos"], subPlotAtt["width"], subPlotAtt["height"]
        axins = ax.inset_axes((xpos, ypos, width, height))
        xlim0, xlim1, ylim0, ylim1 = subPlotAtt["xlim"][0], subPlotAtt["xlim"][1], subPlotAtt["ylim"][0], subPlotAtt["ylim"][1]
        rangeS, rangeE = subPlotAtt["range"][0], subPlotAtt["range"][1]

        for i in range (3):
            axins.plot(time[rangeS: rangeE], (neu[rangeS: rangeE, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)
        # axins.grid(b=False, linestyle='--', color='k', alpha=0.4)

        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        xy = (xlim0,ylim0)
        xy2 = (xlim0,ylim1)
        con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
        ax.add_artist(con)

        xy = (xlim1,ylim0)
        xy2 = (xlim1,ylim1)
        con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                axesA=axins,axesB=ax)
        ax.add_artist(con)

        # 原图中画方框
        tx0 = xlim0
        tx1 = xlim1
        ty0 = ylim0
        ty1 = ylim1
        sx = [tx0,tx1,tx1,tx0,tx0]
        sy = [ty0,ty0,ty1,ty1,ty0]
        ax.plot(sx,sy,"black")
        axins.set_alpha(0)
        axins.set_facecolor((1, 1, 1, 1))
        # axins.ba
        # mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec='k', lw=1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    if attribute["xlim"][0] == attribute["xlim"][1]:
        plt.xlim(time[0], time[-1])
    else:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] == attribute["ylim"][1]:
        pass
    else:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.175, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def ploterror(time, neu, save, attribute, isSubplot):
    if neu.shape[0] < 1:
        return
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
         0: [(0 / 255), (141 / 255), (0 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
    # print(save)
    for i in range(neu.shape[1]):

        # ax = plt.subplot(3, 1, i + 1)
        # plt.tight_layout()
        # if i != 2:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        plt.grid(linestyle='--', color='k', alpha=0.4)
        direc_ = neu[:, i]
        plt.plot(time, (neu[:, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)

        print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel(attribute['ylabel'], labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    if isSubplot:
        subPlotAtt = attribute["subplot"]
        xpos, ypos, width, height = subPlotAtt["xpos"], subPlotAtt["ypos"], subPlotAtt["width"], subPlotAtt["height"]
        axins = ax.inset_axes((xpos, ypos, width, height))
        rangeS, rangeE = subPlotAtt["range"][0], subPlotAtt["range"][1]
        for i in range (3):
            axins.plot(time[rangeS: rangeE], (neu[rangeS: rangeE, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)
        # axins.grid(b=False, linestyle='--', color='k', alpha=0.4)
        
        ylimS, ylimE = subPlotAtt["ylim"][0], subPlotAtt["ylim"][1]
        axins.set_xlim(subPlotAtt["xlim"][0], subPlotAtt["xlim"][1])
        axins.set_ylim(ylimS, ylimE)
        # mark_inset()
        loc1, loc2 = subPlotAtt["loc"][0], subPlotAtt["loc"][1]
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec='k', lw=1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    
    if attribute["scientific"]:
        # scientific expression
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,0)) 
        ax.yaxis.set_major_formatter(formatter)

    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotTraj(time, trajs={ }, save='./'):
    """plot trajectories
       
    Args:
        trajs (list, optional): _description_. Defaults to [].
        each one is a traj(in enu-frame)
    """
    plt.figure(figsize=(5.0393701, 5.0393701))
    # plt.subplots_adjust(hspace=0.5,wspace=0.6)
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = plt.gca()
    ax.set_aspect(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    gs = GridSpec(10, 20)
    # plt.subplot(gs[:8,:])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.grid(linestyle='-', color=grey, alpha=0.5, linewidth=1)
    i = 0
    for key in trajs.keys():
        plt.plot(trajs[key][:, 0], trajs[key][:, 2], label=key, color=color[i])
        i += 1
    font_={
      'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold'
    }
    plt.xticks(size = 11, fontproperties='Cambria')
    plt.yticks(size = 11, fontproperties='Cambria')
    legend = plt.legend(loc='upper right', fontsize = 11, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.0209, 1.1), fancybox=False)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.91, wspace=0.01, hspace=0.1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    xmajorLocator  = MultipleLocator(30)
    ymajorLocator  = MultipleLocator(10)
    plt.ylim(-100, 150)
    plt.xlim(-225, 25)
    plt.xticks([-225, -175, -125, -75, -25, 25])
    plt.ylabel("y (m)", fontdict=font_)
    plt.xlabel("x (m)", fontdict=font_)
    # plt.subplot(gs[9:, 1:])
    # ax = plt.gca()
    # # ax.set_aspect(1)
    # i = 0
    # for key in trajs.keys():
    #     plt.plot(time, trajs[key][:, 2], label=key, color=color[i])
    #     i += 1
    # plt.xlim(0, 70)
    # plt.ylim(-60, 60)
    # plt.grid(b=False, linestyle='--', color='k', alpha=0.4)
    # # ax.xaxis.set_major_locator(x_major_locator)
    # # ax.yaxis.set_major_locator(y_major_locator)
    # plt.ylabel("Height(m)", fontdict=font_)
    # plt.xlabel("Epoch(s)", fontdict=font_)
    # plt.margins(x=0, y=0.00)
    plt.savefig(save, transparent=True)
    plt.show()


def plotBox(neu):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='-', color=grey, alpha=0.5, linewidth=1)
    
    test = []
    # test.
    for key in neu.keys():
        test.append(neu[key])
    plt.boxplot(test, showfliers=True, 
                labels=neu.keys(), whis=1.5, flierprops={'marker':'+', 'markeredgecolor': 'red'},
                medianprops={'color': 'green'}
                )
        
    plt.ylim(0, 10)
    plt.margins(x=0, y=0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.xticks(size = 12, fontproperties='Cambria')
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.legend()
    # plt.savefig(r"D:\文件\learn\01-本科\毕业设计\06-实验结果\fig\box.svg",transparent=True)
    plt.show()
    # pass

def plotSatNum(satNum, save):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='-', color=grey, alpha=0.5, linewidth=1)
    
    plt.scatter(range(0, satNum.shape[0]), satNum, linewidths=1.5, c=color[1],s=5, label="G+C+E")
    # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # ax = legend.get_frame()
    # ax.set_alpha(1)
    # ax.set_facecolor('none')

    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    ax = plt.gca()
    xmajorLocator  = MultipleLocator(1500)
    ymajorLocator  = MultipleLocator(5)

    ax.xaxis.set_major_locator(xmajorLocator) 

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    
    label = []
    test = plt.xticks(size = 12, fontproperties='Cambria')
    for i in range(0, len(test[0])):
        label.append("{:.1f}".format(test[0][i] / 60 / 10))
    plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    plt.xlim(0, 7500)
    plt.ylim(0, 30)

    plt.ylabel("Number of Satellite", labelpad=3, fontsize = 13, fontdict=font)
    plt.xlabel("Epoch (min)", fontdict=font)

    plt.savefig(save, transparent=True)
    plt.show()
    

def plotPdop(pdop, save):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    
    plt.scatter(range(0, pdop.shape[0]), pdop, linewidths=1.5, c=color[2],s=4.5, marker=",")
    # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # ax = legend.get_frame()
    # ax.set_alpha(1)
    # ax.set_facecolor('none')

    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    ax = plt.gca()
    # plt.xlim(0, 00)
    xmajorLocator  = MultipleLocator(1500)
    ymajorLocator  = MultipleLocator(5)

    ax.xaxis.set_major_locator(xmajorLocator) 

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    label = []
    test = plt.xticks(size = 12, fontproperties='Cambria')
    for i in range(0, len(test[0])):
        label.append("{:.1f}".format(test[0][i] / 60 / 10))
    plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    plt.xlim(0, 7500)

    plt.ylabel("PDOP", labelpad=3, fontsize = 13, fontdict=font)
    plt.xlabel("Epoch (min)", fontdict=font)
    plt.ylim(0, 8)
    plt.margins(x=0, y=0)

    plt.savefig(save, transparent=True)
    plt.show()
    
def plotCDF(cdf_dict = {}, save=""):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    plt.margins(x=0, y=0)
    
    i = 1
    for key in cdf_dict.keys():
        cdf_dict[key] = np.sort(np.fabs(cdf_dict[key]))
        plt.plot(cdf_dict[key], np.array(range(0, cdf_dict[key].shape[0])) / cdf_dict[key].shape[0], label=key, color=color[i], linewidth=2)
        i+=1

    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.155), fancybox=False)
    
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.89, wspace=0.01, hspace=0.1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.xlabel("Error (m)", fontdict=font)
    plt.ylabel("Probability", labelpad=3, fontsize = 13, fontdict=font)
    plt.ylim(0, 1)
    plt.xlim(0, 30)
    
    plt.savefig(save, transparent=True)
    plt.show()
    
def plotBar(bar_dict={}, save="./"):
    # 目前仅实现了三个为一组的柱状图
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    
    x = np.arange(len(bar_dict.keys()))
    width = 0.25
    data = []
    for key in bar_dict.keys():
        data.append(bar_dict[key])
    data = np.array(data)
    
    labels = ["E", "N", "U"]
    print(data)
    print(x)
    i, j = -1, 0
    for key in bar_dict.keys():
        plt.bar(x + i * width, data[:, j], width, label=labels[j], color=color[j])
        i += 1
        j += 1
    plt.xticks(x, labels=bar_dict.keys(), size = 12, fontproperties='Cambria')
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.155), fancybox=False)
    
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.1, top=0.89, wspace=0.01, hspace=0.1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    # plt.xticks(size = 12, fontproperties='Cambria')
    plt.ylabel("Error (m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.ylim(0, 14)
    
    plt.savefig(save, transparent=True)
    
    plt.show()
    

def plotErrorWithCov(time, data, std, direc, save):
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5.0393701, 3.4645669))
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    # direc_ = direc
    plt.plot(time, data , ls="-", color="b", label=direc, linewidth=2)#, marker=marker[i], markersize=4)

    plt.plot(time, std * 3, linewidth=1.5, ls="--", color="r", label="3-sigma")
    plt.plot(time, -std * 3, linewidth=1.5, ls="--", color="r")

    # print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    plt.xlim(0, 60)
    plt.ylim(-8, 8)
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def CompareCov(time_data1, time_data2, data1, data2, label1, label2, save):
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5.0393701, 3.4645669))
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    # direc_ = direc
    plt.plot(time_data1, data1, ls="-", color="b", label=label1, linewidth=2)#, marker=marker[i], markersize=4)

    plt.plot(time_data2, data2, ls="-", color="r", label=label2, linewidth=2)

    # print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    plt.xlim(0, 60)
    # plt.ylim(-8, 8)
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotPointsWithTraj(trajs=[], points=[], save="./"):
    plt.ioff()
    plt.close()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = plt.figure(figsize=(6, 6)).add_subplot(projection='3d')
    ax.view_init(elev=41, azim=56, roll=0)
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    # plt.grid(b=False, linestyle='--', color='k', alpha=0.5)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color[3], s=30)
    ax.plot(trajs[:, 0], trajs[:, 1], trajs[:, 2], linewidth=4, c=color[2])
    # ax.grid(False)
    ax.ticklabel_format(style='sci', axis='x')
    ax.ticklabel_format(style='sci', axis='y')
    ax.ticklabel_format(style='sci', axis='z')

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.set_zlim(0, 10)
    ax.set_xticks([-17.5, -12.5, -7.5, -2.5, 2.5, 7.5])
    ax.set_yticks([-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])
    ax.set_zticks([-10, -5, 0, 5, 10])

    ax.tick_params(axis='y',
                 labelsize=11) 
    ax.tick_params(axis='x',
                 labelsize=11) 
    ax.tick_params(axis='z',
                 labelsize=11)
    plt.ylim(-12.5, 12.5)
    plt.xlim(-17.5, 7.5)
    # plt.zlim(-2, 2)
    ax.set_zlim(-10, 10)

    ax.set_xlabel("x (m)", fontdict=font)
    ax.set_ylabel("y (m)", fontdict=font)
    ax.set_zlabel("z (m)", fontdict=font)

    ax = plt.gca()
    ax.set_aspect('equal')

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # plt.yticks([])
    # plt.xticks([])

    # plt.figure(2)
    # plt.plot(trajs[:, 0], trajs[:, 1])
    # ax = plt.gca()
    # ax.set_aspect(1)


    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))

    # plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def plotPointsWithTraj2D(trajs=[], points=[], save="./"):
    plt.ioff()
    plt.close()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = plt.figure(figsize=(3, 3))
    # ax.view_init(elev=45, azim=60, roll=0)
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    # plt.grid(b=False, linestyle='--', color='k', alpha=0.5)
    plt.plot(trajs[:, 0], trajs[:, 1], linewidth=4, c=color[2], alpha=0.8)

    plt.scatter(points[:, 0], points[:, 1], color=color[3], s=7)
    # ax.grid(False)
    ax = plt.gca()
    plt.ylabel("y [m]", labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.set_zlim(0, 10)
    # ax.set_xticks([-15, -10, -5, 0, 5])
    # ax.set_yticks([-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])
    # ax.set_zticks([-2, 0, 2])
    ax.set_xlabel("x (m)", fontdict=font)
    ax.set_ylabel("y (m)", fontdict=font)
    # ax.set_zlabel("z (m)", fontdict=font)
    # plt.grid(linestyle='--', color='k', alpha=0.4)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.xlabel("m", fontdict=font)
    plt.ylim(-15, 15)
    plt.xlim(-20, 10)

    ax = plt.gca()
    ax.set_aspect('equal')

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.margins(x=0, y=0)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # plt.yticks([])
    # plt.xticks([])
    # plt.ylim(-12.5, 12.5)
    # plt.xlim(-15, 5)
    # plt.zlim(-2, 2)
    # ax.set_zlim(-2, 2)
    # plt.figure(2)
    # plt.plot(trajs[:, 0], trajs[:, 1])
    # ax = plt.gca()
    # ax.set_aspect(1)


    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))

    # plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def plotLogY(time, y, save, attribute, isSubplot):
    # if neu.shape[0] < 1:
    #     return
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(255 / 255), (204 / 255), (102 / 255)],  # blue
         0: [(0 / 255), (141 / 255), (0 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']
    if len(direc) != 0:
        for i in range(len(y)):
            plt.semilogy(time, y[i], ls="-", color=color[i], label=direc[i], linewidth=2)
    else:
        for i in range(len(y)):
            plt.semilogy(time, y[i], ls="-", color=color[i], linewidth=2)
    plt.ylabel(attribute['ylabel'], labelpad=3, fontdict=font)
    plt.yticks(size = 11)
    plt.xticks(size = 11)
    # plt.ylim(-3, 3)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)
    plt.grid(linestyle='-', color='k', alpha=0.2)
    if len(direc) != 0:
        legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
        ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    
    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    # ax.get_yticklabels().set_color("red")
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()



def plotLogY_Twiny(time, y1, y2, save, attribute, isSubplot):
    # if neu.shape[0] < 1:
    #     return
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(0 / 255), (148 / 255), (255 / 255)],  # blue
         0: [(0 / 255), (141 / 255), (0 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']
    if len(direc) != 0:
        for i in range(len(y1)):
            line1 = ax.semilogy(time, y1[i], ls="-", color=color[i], label=direc[i], linewidth=2)
    else:
        for i in range(len(y1)):
            line1 = ax.semilogy(time, y1[i], ls="-", color=color[i], linewidth=2)
    font_tmp={
    #   'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold',
      'color': color[0]
    }

    plt.ylabel(attribute['y1label'], labelpad=3, fontdict=font)
    plt.yticks(size = 11)
    plt.xticks(size = 11)
    # plt.ylim(-3, 3)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)
    plt.grid(linestyle='-', color='k', alpha=0.2)
    ax = plt.gca()
    ax.set_xlabel(attribute['xlabel'], fontdict=font)

    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["y1lim"][0] != attribute["y1lim"][1]:
        ax.set_ylim(attribute["y1lim"][0], attribute["y1lim"][1])
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_color(color[0])
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_color(color[0])
    ax.tick_params(which='minor',colors=color[0])
    ax.tick_params(axis='y', color=color[0])

    ax2 = ax.twinx()
    line2 = ax2.semilogy(time, y2[0], ls="-", color=color[1], label=direc[1], linewidth=2)
    if attribute["y2lim"][0] != attribute["y2lim"][1]:
        plt.ylim(attribute["y2lim"][0], attribute["y2lim"][1])

    font_tmp={
    #   'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold',
      'color': color[1]
    }

    plt.ylabel(attribute['y2label'], labelpad=3, fontdict=font)
    plt.yticks(size = 11)
    plt.xticks(size = 11)

    ax2.spines['right'].set_color(color[1])
    for i in range(len(ax2.get_yticklabels())):
        ax2.get_yticklabels()[i].set_color(color[1])
    ax2.tick_params(which='minor',colors=color[1])
    ax2.tick_params(axis='y', color=color[1])

    lines = line1 + line2  
    labels = [h.get_label() for h in lines]  
    legend = plt.legend(lines, labels, loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    ax = legend.get_frame()
    # if len(direc) != 0:
    #     legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlabel(attribute["xlabel"], fontdict=font)
    # ax.get_yticklabels().set_color("red")
    plt.subplots_adjust(left=0.16, right=0.84, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.savefig(save, transparent=True)
    plt.show()


def plotTrajWithError2D(gt_pos, error, attribute, save):
    plt.ioff()
    plt.close()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(figsize=(3, 3))
    xlim, ylim = attribute['xlim'], attribute['ylim']
    ax = plt.gca()
    if xlim[0] != xlim[1] and ylim[0] != ylim[1]:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])    
        plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
        plt.yticks(range(ylim[0], ylim[1] + 1, ylim[2]), size = 12, fontproperties='Cambria')
        plt.xticks(range(xlim[0], xlim[1] + 1, xlim[2]), size = 12, fontproperties='Cambria')
    if attribute["scientific"]:
        # scientific expression
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,0)) 
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
    points = np.array([gt_pos[:, 0], gt_pos[:, 2]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    barrange = attribute['barrange']
    norm = mcolors.LogNorm(barrange[0], barrange[1])
    # norm = plt.Normalize(error.min(), error.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(error)
    lc.set_linewidth(4) 
    line = ax.add_collection(lc) 
    # bar = fig.colorbar(line, ax=ax, orientation='horizontal', fraction=attribute["barfraction"], pad=0.04, shrink=1.0)
    # bar.ax.tick_params(labelsize = 11)
    # bar.set_label(attribute['barlabel'], fontdict=font1)
    plt.xticks(size = 11)
    plt.yticks(size = 11)
    ax = plt.gca()
    # plt.ylabel("y [m]", labelpad=3, fontsize = 13, fontdict=font)
    # ax.set_xticks(ax.get_xticks(), size = 15, fontproperties='Calibri')

    # ax.set_xlabel(attribute["xlabel"], fontdict=font)
    # ax.set_ylabel(attribute["ylabel"], fontdict=font)
    # ax.set_zlabel("z (m)", fontdict=font)
    plt.grid(linestyle='-', color='k', alpha=0.2, linewidth=1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    # # ax.set_xlabel("x (m)", fontdict=font)
    # # ax.set_ylabel("y (m)", fontdict=font)
    # # ax.set_zlabel("z (m)", fontdict=font)

    ax = plt.gca()
    # ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
    ax.set_aspect(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    # ax.set_xticks(size = 15, fontproperties='Calibri')
    # ax.set_yticks(size = 15, fontproperties='Calibri')

    plt.margins(x=0, y=0)


    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9, wspace=0.01, hspace=0.1)
    # plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotCompareDot(error_dict = {}, attribute = {}, save="./"):
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (146 / 255), (0 / 255)],  # red
         1: [(0 / 255), (141 / 255), (0 / 255)],  # blue
         0: [(229 / 255), (8 / 255), (106 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    # color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
    #      2: [(229 / 255), (87 / 255), (9 / 255)],  # red
    #      1: [(28 / 255), (128 / 255), (65 / 255)],  # blue
    #      0: [(80 / 255), (29 / 255), (138 / 255)],
    #      4: [(20 / 255), (169 / 255), (89 / 255)],
    #      5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(3.4645669, 3.4645669))
    direc = attribute['legend']

    min_x, min_y = attribute["xlim"][0], attribute["ylim"][0]
    max_x, max_y = attribute["xlim"][1], attribute["ylim"][1]
    plt.plot([min_x, max_x], [min_y, max_y], c="black", ls="-", linewidth=1, alpha=0.4)
    i = 0
    for label, compare_np in error_dict.items():
        plt.scatter(compare_np[:, 0], compare_np[:, 1], marker="o", label=label, alpha=0.7, c=color[i], linewidths=0.6, edgecolors="black", s=60)
        i += 1

    # if len(direc) != 0:
    #     for i in range(len(y)):
    #         plt.semilogy(time, y[i], ls="-", color=color[i], label=direc[i], linewidth=2)
    # else:
    #     for i in range(len(y)):
    #         plt.semilogy(time, y[i], ls="-", color=color[i], linewidth=2)
    plt.ylabel(attribute['ylabel'], labelpad=3, fontdict=font)
    # plt.ylim(-3, 3)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)
    plt.grid(linestyle='-', color='k', alpha=0.2)

    # if len(direc) != 0:
    #     legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    #     ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    plt.xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5], size = 11)
    plt.yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5], size = 11)

    ax = plt.gca()
    
    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax = plt.gca()
    # ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
    ax.set_aspect(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def plotLineWithDot(xticks, error_dict = {}, attribute = {}, save="./"):
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(0 / 255), (148 / 255), (255 / 255)],  # blue
         0: [(0 / 255), (141 / 255), (0 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    # color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
    #      2: [(229 / 255), (87 / 255), (9 / 255)],  # red
    #      1: [(28 / 255), (128 / 255), (65 / 255)],  # blue
    #      0: [(80 / 255), (29 / 255), (138 / 255)],
    #      4: [(20 / 255), (169 / 255), (89 / 255)],
    #      5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']

    min_x, min_y = attribute["xlim"][0], attribute["ylim"][0]
    max_x, max_y = attribute["xlim"][1], attribute["ylim"][1]
    for key, value in error_dict.items():
        # plt.plot(xticks, value, ls="-", linewidth=1, marker='o', color=color[i])
        plt.plot(xticks, value, ls="-", linewidth=1, marker='o')
    # if len(direc) != 0:
    #     for i in range(len(y)):
    #         plt.semilogy(time, y[i], ls="-", color=color[i], label=direc[i], linewidth=2)
    # else:
    #     for i in range(len(y)):
    #         plt.semilogy(time, y[i], ls="-", color=color[i], linewidth=2)
    plt.ylabel(attribute['ylabel'], labelpad=3, fontdict=font)
    # plt.ylim(-3, 3)
    # plt.subplots_adjust(top=1)
    # plt.margins(x=0, y=0)
    plt.grid(linestyle='-', color='k', alpha=0.2)

    # if len(direc) != 0:
    #     legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    #     ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    plt.xticks(size = 11)
    plt.yticks(size = 11)

    ax = plt.gca()
    
    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax = plt.gca()
    # ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
    # ax.set_aspect(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()
