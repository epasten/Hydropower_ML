import pandas as pd
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Read the dataframes from the LSTM results

whalley_tascal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/whalley_results_tascal.pkl')
whalley_precipcal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/whalley_results_precipcal.pkl')

beddgelert_tascal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/beddgelert_results_tascal.pkl')
beddgelert_precipcal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/beddgelert_results_precipcal.pkl')

eynsham_tascal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/eynsham_results_tascal.pkl')
eynsham_precipcal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/eynsham_results_precipcal.pkl')

rothsbury_tascal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/rothsbury_results_tascal.pkl')
rothsbury_precipcal = pd.read_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/rothsbury_results_precipcal.pkl')

# plotting NSE vc. KGE
plt.figure(figsize=(10,10),dpi=200)
color_plot= ['r','g']
legend_elements = [Line2D([0], [0], marker='s',color='k', label='Whalley',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='o', color='k', label='Beddgelert',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='d', color='k', label='Eynsham',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='v', color='k', label='Rothsbury',
       markerfacecolor='k', markersize=10,linestyle='None'),
Patch(facecolor='r', edgecolor='r', label='Control'),
Patch(facecolor='g', edgecolor='g', label='Evaluation'),
Patch(facecolor='k', edgecolor='k', label='Precipitation-based'),
Patch(facecolor='white', edgecolor='k', label='Temperature-based')]
for i in [0,1]:# 0 is the control period and 1 is the evalaution period
    plt.scatter(whalley_tascal.NSE[i],whalley_tascal.KGE[i],s=80,edgecolors=color_plot[i],marker='s',facecolors='none')
    plt.scatter(whalley_precipcal.NSE[i],whalley_precipcal.KGE[i],s=80,c=color_plot[i],marker='s')
    plt.scatter(beddgelert_tascal.NSE[i],beddgelert_tascal.KGE[i],s=80,edgecolors=color_plot[i],marker='o',facecolors='none')
    plt.scatter(beddgelert_precipcal.NSE[i],beddgelert_precipcal.KGE[i],s=80,c=color_plot[i],marker='o')
    plt.scatter(eynsham_tascal.NSE[i],eynsham_tascal.KGE[i],s=80,edgecolors=color_plot[i],marker='d',facecolors='none')
    plt.scatter(eynsham_precipcal.NSE[i],eynsham_precipcal.KGE[i],s=80,c=color_plot[i],marker='d')
    plt.scatter(rothsbury_tascal.NSE[i],rothsbury_tascal.KGE[i],s=80,edgecolors=color_plot[i],marker='v',facecolors='none')
    plt.scatter(rothsbury_precipcal.NSE[i],rothsbury_precipcal.KGE[i],s=80,c=color_plot[i],marker='v')
plt.title('LSTM results - KGE Vs. NSE')
plt.grid('True')
plt.legend(handles=legend_elements, loc='lower right',fontsize=10) #,bbox_to_anchor=(1.4,0.2))   
plt.ylabel('KGE')
plt.xlabel('NSE')
plt.ylim((0,1))
plt.xlim((0,1))
plt.savefig('C:/Users/erpasten/Documents/UEF/Hydropower/figures/lstm_NSE_KGE.png',dpi=200)

# plotting biases in discharge percentiles
plt.figure(figsize=(10,10),dpi=200)
color_plot= ['r','g']
legend_elements = [Line2D([0], [0], marker='s',color='k', label='Whalley',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='o', color='k', label='Beddgelert',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='d', color='k', label='Eynsham',
       markerfacecolor='k', markersize=10,linestyle='None'),
Line2D([0], [0], marker='v', color='k', label='Rothsbury',
       markerfacecolor='k', markersize=10,linestyle='None'),
Patch(facecolor='r', edgecolor='r', label='Control'),
Patch(facecolor='g', edgecolor='g', label='Evaluation'),
Patch(facecolor='k', edgecolor='k', label='Precipitation-based'),
Patch(facecolor='white', edgecolor='k', label='Temperature-based')]
for i in [0,1]:# 0 is the control period and 1 is the evalaution period
    plt.scatter(whalley_tascal.Q5_Bias[i],whalley_tascal.Q95_Bias[i],s=80,edgecolors=color_plot[i],marker='s',facecolors='none')
    plt.scatter(whalley_precipcal.Q5_Bias[i],whalley_precipcal.Q95_Bias[i],s=80,c=color_plot[i],marker='s')
    plt.scatter(beddgelert_tascal.Q5_Bias[i],beddgelert_tascal.Q95_Bias[i],s=80,edgecolors=color_plot[i],marker='o',facecolors='none')
    plt.scatter(beddgelert_precipcal.Q5_Bias[i],beddgelert_precipcal.Q95_Bias[i],s=80,c=color_plot[i],marker='o')
    plt.scatter(eynsham_tascal.Q5_Bias[i],eynsham_tascal.Q95_Bias[i],s=80,edgecolors=color_plot[i],marker='d',facecolors='none')
    plt.scatter(eynsham_precipcal.Q5_Bias[i],eynsham_precipcal.Q95_Bias[i],s=80,c=color_plot[i],marker='d')
    plt.scatter(rothsbury_tascal.Q5_Bias[i],rothsbury_tascal.Q95_Bias[i],s=80,edgecolors=color_plot[i],marker='v',facecolors='none')
    plt.scatter(rothsbury_precipcal.Q5_Bias[i],rothsbury_precipcal.Q95_Bias[i],s=80,c=color_plot[i],marker='v')
plt.title('LSTM results - Bias in Q5 Vs. Bias in Q95')
plt.grid('True')
plt.legend(handles=legend_elements, loc='center left',fontsize=10) #,bbox_to_anchor=(1.4,0.2))   
plt.ylabel('Q95 bias in m3/s (Simulation - Observation)')
plt.xlabel('Q5 bias in m3/s (Simulation - Observation)')
#plt.ylim((0,1))
#plt.xlim((0,1))
plt.savefig('C:/Users/erpasten/Documents/UEF/Hydropower/figures/lstm_Q5_Q95.png',dpi=200)
