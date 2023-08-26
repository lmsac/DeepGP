import pandas as pd
from pathlib import Path
# ----------------------- args ------------------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern",type=str,default='*_rt_1st.csv')
    parser.add_argument("--Cali_csv",type=str,default='/remote-home/yxwang/test/DeepGP_code/data/mouse/PXD005411/PXD005411_MouseBrain_rt_1st.csv')
    parser.add_argument('--folder_path', type=str, 
                        default="/remote-home/yxwang/test/DeepGP_code/data/mouse/", 
                        help='the folder path for the training data')
    parser.add_argument('--output_name', type=str, 
                        default="All_adjust_irt.csv", 
                        help='the output filename for the irt peptides')
    args = parser.parse_args()
    return args
args=parsering()
folder_path = args.folder_path
output_name=args.output_name
pattern=args.pattern
Cali=args.Cali_csv
# ----------------------- irt peptide ------------------------------#
traincsv=pd.read_csv(Cali)
traincsv["y"]=traincsv["RT"]
traincsv.sort_values(by='y',ascending=True,inplace=True)
traincsv=traincsv[["y","iden_pep"]]
print(traincsv)
print("the number of irt peptide", len (traincsv[["iden_pep"]].drop_duplicates()))
# ----------------------- testcsv ------------------------------#
def dev(row):
    return abs(row["rt"]-row["y"])/row["rt"]
import folder_walk
trainpathcsv_list=folder_walk.trainpathcsv_list(folder_path=folder_path,pattern=pattern)
print(f"Please check trainpathcsv_list {trainpathcsv_list}. The {len(trainpathcsv_list)} files contains all the training data!")
testdatacsv=pd.DataFrame()
for x in  trainpathcsv_list:
        train=pd.read_csv(x)
        # ipdb.set_trace()
        train=train[['GlySpec', 'charge_mgf', 'RT_mgf', 'Peptide', 'Mod', 'PlausibleStruct','GlySite', 'iden_pep', 'TotalFDR', 'PrecursorMZ']]
        train.columns=['GlySpec', 'Charge', 'RT', 'Peptide', 'Mod', 'PlausibleStruct','GlySite', 'iden_pep', 'TotalFDR', 'PrecursorMZ']
        testdatacsv=pd.concat([testdatacsv,train])
        print(f"with the addition of {x}, the combined file contains {len(testdatacsv)} lines")
testdatacsv["run"]=testdatacsv["GlySpec"].apply(lambda x: x.split("-")[0])
testdatacsv.sort_values(by='TotalFDR',ascending=True,inplace=True)
testdatacsv.drop_duplicates(subset=['iden_pep',"run"],inplace=True)
testdatacsv.reset_index(drop=True,inplace=True)
testdatacsv.rename(columns={'RT': 'rt'}, inplace=True)
testdatacsv=pd.merge(testdatacsv,traincsv,on=["iden_pep"],how="left")
testdatacsv["dev"]=testdatacsv.apply(dev,axis=1)
print(testdatacsv[["rt","y","dev"]].dropna())
testdatacsv.sort_values(by='run',ascending=True,inplace=True)
testdatacsv.reset_index(drop=True,inplace=True)
# ----------------------- Regression ------------------------------#
import copy
import numpy as np
class RetentionTimeCalibrator():
    def __init__(self, model='interpolate', 
                 smooth='lowess'):
        from scipy.interpolate import interp1d
        def interpolate(x, y, new_x):  
            x, index = np.unique(x, return_index=True)
            y = y[index] 
            interp = interp1d(x, y, fill_value='extrapolate',kind="linear")
            return interp(new_x)
        self.model_func = interpolate
        if smooth == 'savgol':
            from scipy.signal import savgol_filter
            def savgol(x, y, **kwargs):
                smooth_args = {
                    'window_length': 7, 
                    'polyorder': 1
                }
                smooth_args.update(kwargs)
                x = savgol_filter(
                    x, **smooth_args
                )
                return x, y
            self.smooth_func = savgol

        elif smooth == 'lowess':
            import statsmodels.api as sm
            
            def lowess(x, y, **kwargs):
                frac = kwargs.get('frac', 0.2) #0.3
                it = kwargs.get('it', 2)
                r = sm.nonparametric.lowess(y, x, frac=frac,it=it)
                # https://github.com/statsmodels/statsmodels/issues/2449
                if any(np.isnan(r[:, 1])):
                    data = pd.DataFrame.from_dict({'x': x, 'y': y}) \
                        .groupby(x).mean()
                    x = data['x']
                    y = data['y']
                    r = sm.nonparametric.lowess(y, x,frac=frac,it=it)
                return r[:, 0], r[:, 1]
            self.smooth_func = lowess
        
    def calculate_rt(self, data):  
        xy=data[["rt","y","dev"]].dropna()
        x=xy["rt"]
        y=xy["y"]
        if self.smooth_func is not None:
            x, y = self.smooth_func(x, y)
        
        if any(map(lambda x: not x > 0, x)):
            raise ValueError(x)
        if any(map(lambda x: not x > 0, y)):
            raise ValueError(y)  
        y_new = self.model_func(x, y, data['rt'].values)
        if min(y_new)<0:
            print("Alert: the ajusted min rt is below zero!")
        return y_new
        
            
    def calibrate_rt(self, assay_data, multiple_runs=False, inplace=False, return_data=False):
            
        if not multiple_runs:                      
            rt_new = self.calculate_rt(assay_data)               
            
        else:
            rt_new = pd.Series(None, index=assay_data.index)
            for run, data in assay_data.groupby(by=['run'], group_keys=False):
                print(run)
                print(data)
                # ipdb.set_trace()
                rt_new[data.index] = self.calculate_rt(data) 
        assay_data.rename(columns={'rt': 'rt_old'}, inplace=True)
        assay_data['rt_new'] = rt_new    
        assay_data=assay_data[['rt_old', 'iden_pep', 'y', 'rt_new',"run"]]
        assay_data.rename(columns={'y': 'rt_reference'}, inplace=True)
        return assay_data
            
calibrator = RetentionTimeCalibrator(
    model="interpolate", 
    smooth="lowess")
# ipdb.set_trace()
rt_data = calibrator.calibrate_rt(
    testdatacsv,
    multiple_runs=True, 
    inplace=False, 
    return_data=True
)
rt_data.to_csv(folder_path+output_name,index=False)

# ----------------------- rt plot ------------------------------#
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
# ipdb.set_trace()

def plot_rt_calibration(rt_data, pdf_path):
    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")
        
    with PdfPages(pdf_path) as pdf:
        groups = rt_data.groupby(['run'])
        # import ipdb
        # ipdb.set_trace() 
        if len(groups) > 1:            
            for idx, (run, data) in enumerate(groups):
                # ipdb.set_trace()
                if idx % 6 == 0:
                    plt.figure(figsize=(10, 15))
                    plt.subplots_adjust(hspace=.75)
                
                plt.subplot(321 + idx % 6)                            
                        
                plt.scatter(data.rt_old, data.rt_new, marker='.',s=3)
                plt.scatter(data.rt_old, data.rt_reference, color='red', marker='D',s=1,alpha=0.8)
                plt.xlabel('Raw RT')
                plt.ylabel('Calibrated RT')
                plt.title(run)
                # ipdb.set_trace()
                if idx % 6 == 5 or idx == len(groups) - 1:            
                    pdf.savefig()
                    plt.close()
                    
        else:
            plt.figure(figsize=(10, 10))
            
            plt.scatter(rt_data.rt_old, rt_data.rt_new, marker='.')
            plt.scatter(rt_data.rt_old, rt_data.rt_reference, color='red', marker='D')
            plt.xlabel('Raw RT')
            plt.ylabel('Calibrated RT')
            plt.title(rt_data['run'][0])
                
            pdf.savefig()
            plt.close()
pdf_path=folder_path+"RT_calibration_plot.pdf"
plot_rt_calibration(rt_data,pdf_path)

# ----------------------- irt for each dataset ------------------------------#
for k in trainpathcsv_list:
    traincsv=pd.read_csv(k)
    # ipdb.set_trace()
    traincsv.rename(columns={'RT': 'rt_old'}, inplace=True)
    traincsv["run"]=traincsv["GlySpec"].apply(lambda x: x.split("-")[0])
    traincsv=pd.merge(traincsv,rt_data,on=["run","iden_pep"],how="left")
    # ipdb.set_trace()
    print(traincsv[["rt_old_x","rt_old_y","rt_new"]])
    traincsv.rename(columns={'rt_new': 'RT'}, inplace=True)
    traincsv=traincsv[['GlySpec', 'Charge', 'RT', 'Peptide', 'Mod', 'PlausibleStruct',
        'GlySite', 'iden_pep', 'TotalFDR', 'PrecursorMZ', 'ions' ]]
    traincsv.to_csv(k[:-4]+"_adjust.csv",index=False)
    print(traincsv["RT"])