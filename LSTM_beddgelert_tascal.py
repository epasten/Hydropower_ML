# Following Koch example, applied to Beddgelert using calibration based on temperature
import pandas as pd
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt


# split a multivariate sequence into samples
# data include: date, Q, P, PET, Ta, month
def split_sequences(sequences, n_steps): # in line 53 , the sequence input is defined as data_norm (normalized data)
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps # n_steps is a perameter defined for LSTM
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, 1:], sequences[end_ix-1, 0] # seq_x is the climate variables (P,PET,Ta), seq_y is the flow output
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
# NSE for perfromance evaluation
def nse(simulation_s, evaluation):
    nse_ = 1 - (np.sum((evaluation - simulation_s) ** 2, axis=0, dtype=np.float64) /
                np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64))

    return nse_

# KGE for perfromance evaluation
def kge(simulation_kge, evaluation_kge):
    correl_coef = np.corrcoef(simulation_kge,evaluation_kge)
    kge_ = 1 - (((correl_coef[0,1]-1)**2)+
                (((np.std(simulation_kge)/np.std(evaluation_kge))-1)**2)+
                (((np.mean(simulation_kge)/np.mean(evaluation_kge))-1)**2))**0.5
    return kge_

# Discharge 95th percentile (high flow) bias

def ptile95(simulation_95, evaluation_95):
    ptile95_ = np.nanpercentile(simulation_95,95)-np.nanpercentile(evaluation_95,95)
    return ptile95_

# Discharge 5th percentile (low flow) bias
def ptile05(simulation_05, evaluation_05):
    ptile05_ = np.nanpercentile(simulation_05,5)-np.nanpercentile(evaluation_05,5)
    return ptile05_

# read data
data=pd.read_excel('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/beddgelert_tascal.xlsx')
data=data.drop('Day',axis=1)
data=data.drop('Month',axis=1)
data=data.drop('Year',axis=1)

data=data.to_numpy()

## ii splits data into test and train: 10 years as training in Koch example
## Here, it is 4 years, inclugind leap year
ii=int(4*365+1)


#normalize data
avg=np.nanmean(data[:ii,:],axis=0)
std=np.nanstd(data[:ii,:],axis=0)
data_norm=np.divide(np.subtract(data,avg),std)

# define parameters for LSTM
n_steps = 10
n_features=3

# convert into input/output
X, y = split_sequences(data_norm, n_steps)

# define model
model = Sequential()
model.add(LSTM(20, activation='relu',return_sequences=True,input_shape=(n_steps, n_features)))
model.add(LSTM(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# train
model.fit(X[0:ii,:,:], y[0:ii], epochs=50, verbose=0)
# make prediction
yhat=np.empty((len(data)-ii))
for i in range(len(yhat)):    
    yhat[i] = model.predict(X[ii+i-n_steps,np.newaxis,:,:], verbose=0)
    print (i)
    
    
# reverse normalization
y=y*std[0]+avg[0]
yhat=yhat*std[0]+avg[0]


nse_control = nse(yhat[0:ii],y[ii-n_steps:ii-n_steps+ii])
y_control = y[ii-n_steps:ii-n_steps+ii]
yhat_control_nan=yhat[0:ii][~np.isnan(y_control)]
y_control_nan = y_control[~np.isnan(y_control)]
nse_control_nan = nse(yhat_control_nan,y_control_nan)

nse_eval = nse(yhat[ii:ii+ii],y[ii-n_steps+ii:ii-n_steps+ii+ii])
y_eval = y[ii-n_steps+ii:ii-n_steps+ii+ii]
yhat_eval_nan=yhat[ii:ii+ii][~np.isnan(y_eval)]
y_eval_nan = y_eval[~np.isnan(y_eval)]
nse_eval_nan = nse(yhat_eval_nan,y_eval_nan)

nse_control = str(round(nse_control,3))
nse_control_nan = str(round(nse_control_nan,3))
nse_eval = str(round(nse_eval,3))
nse_eval_nan = str(round(nse_eval_nan,3))


kge_control = kge(yhat[0:ii],y[ii-n_steps:ii-n_steps+ii])
y_control = y[ii-n_steps:ii-n_steps+ii]
yhat_control_nan=yhat[0:ii][~np.isnan(y_control)]
y_control_nan = y_control[~np.isnan(y_control)]
kge_control_nan = kge(yhat_control_nan,y_control_nan)

kge_eval = kge(yhat[ii:ii+ii],y[ii-n_steps+ii:ii-n_steps+ii+ii])
y_eval = y[ii-n_steps+ii:ii-n_steps+ii+ii]
yhat_eval_nan=yhat[ii:ii+ii][~np.isnan(y_eval)]
y_eval_nan = y_eval[~np.isnan(y_eval)]
kge_eval_nan = kge(yhat_eval_nan,y_eval_nan)

kge_control = str(round(kge_control,3))
kge_control_nan = str(round(kge_control_nan,3))
kge_eval = str(round(kge_eval,3))
kge_eval_nan = str(round(kge_eval_nan,3))

Q5_control = ptile95(yhat[0:ii],y[ii-n_steps:ii-n_steps+ii])
Q5_control = str(round(Q5_control,3))
Q5_eval = ptile95(yhat[ii:ii+ii],y[ii-n_steps+ii:ii-n_steps+ii+ii])
Q5_eval = str(round(Q5_eval,3))

Q95_control = ptile05(yhat[0:ii],y[ii-n_steps:ii-n_steps+ii])
Q95_control = str(round(Q95_control,3))
Q95_eval = ptile05(yhat[ii:ii+ii],y[ii-n_steps+ii:ii-n_steps+ii+ii])
Q95_eval = str(round(Q95_eval,3))

# Results arrangement: Row0 = Control; Row1 = Eval; Column1=NSE; Column2=KGE; Column3=Q5 bias; Column4= Q95 bias
result_bed_tascal =  np.zeros([2,4])
result_bed_tascal[0,0]= nse_control_nan
result_bed_tascal[1,0]= nse_eval_nan
result_bed_tascal[0,1]= kge_control_nan
result_bed_tascal[1,1]= kge_eval_nan
result_bed_tascal[0,2]= Q5_control
result_bed_tascal[1,2]= Q5_eval
result_bed_tascal[0,3]= Q95_control
result_bed_tascal[1,3]= Q95_eval

df = pd.DataFrame({'NSE':result_bed_tascal[:,0],'KGE':result_bed_tascal[:,1],'Q5_Bias':result_bed_tascal[:,2],'Q95_Bias':result_bed_tascal[:,3],'Period':['Control','Evaluation']})
#Saving the dataframe as pickle (keeps the format of the data) and then as csv
df.to_pickle('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/beddgelert_results_tascal.pkl')
df.to_csv('C:/Users/erpasten/Documents/UEF/Hydropower/data/ml/beddgelert_results_tascal.csv',index=False)
print(df)

# plotting
plt.figure(figsize=(12,5))
plt.plot(np.linspace(0,ii-1,num=ii),y[0:ii],'k',label='observed / training')# Training period, First 1461 values
plt.plot(np.linspace(ii,ii+len(yhat),num=len(yhat)),y[ii-n_steps:ii-n_steps+len(yhat)],label='observed')# Observed values of the control/evaluation periods
plt.plot(np.linspace(ii,ii+ii-1,num=ii),yhat[0:ii],label='Control LSTM')# Simulation for the control period
plt.plot(np.linspace(ii+ii-1,(ii+ii-1)+ii-1,num=ii),yhat[ii:len(yhat)],label='Evaluation LSTM')# Simulation for the evaluation period (constrasting clmiate)
plt.ylabel('Q [m3/day]')
plt.legend()
plt.title('Beddgelert Daily discharge - LSTM DSST for temperature')
plt.grid('True')
plt.text(2100,-5,r'NSE:'+nse_control)
plt.text(3500,-5,r'NSE:'+nse_eval)
#plt.ylim((0,50))
#plt.gca().set_ylim(bottom=0)
#plt.xlim(0,10943)
plt.xlabel('Days')
plt.savefig('C:/Users/erpasten/Documents/UEF/Hydropower/figures/beddgelert_lstm_discharge_tascal.png',dpi=400)
    


