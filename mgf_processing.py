import pandas as pd
from pyteomics import mgf
import re
import os
import numpy as np
# import argparse
# import json
import torch
# --------------------------- mgf processing ------------------------------#
def mgf_process(mgfdatafold:str,sourceorign:str):
    """Turn mgf to json.

    Args:
    sourceorign: MsConvert or pGlyco3
    
    """
    jsonfold = os.path.join(mgfdatafold, "json")
    if os.path.exists(jsonfold):
        0
    else:
        os.mkdir(jsonfold)
    for filename in os.listdir(mgfdatafold):
        if filename.endswith("mgf"):
            mgfname=os.path.join(mgfdatafold,filename)
            print(mgfname)
            mgfdata=pd.DataFrame(columns=["SourceFile","Spectrum","RT_mgf","pepmass_mgf","charge_mgf","intensity","mz"])
            down=0
            for k,spectrum in enumerate(mgf.read(mgfname)):
                # ipdb.set_trace()
                params = spectrum.get('params')
                title = params.get('title')
                if sourceorign == "MsConvert":
                    namere = re.compile(r'File:"(.*?)"', re.S)  # 最小匹配
                    cNre = re.compile(r'controllerNumber=(\d+)', re.S)
                    scanre = re.compile(r'scan=(.*?)"', re.S)
                    sourcefile=namere.findall(title)[0]
                    cNumber=cNre.findall(title)[0]
                    scan=scanre.findall(title)[0]
                    Spectrum = str(scan)
                    RT_mgf=params.get('rtinseconds')
                    pepmass_mgf=params.get('pepmass')[0]
                    charge_mgf=params.get('charge')[0]
                    # ipdb.set_trace()
                if sourceorign=="pGlyco3":
                    # import ipdb
                    # ipdb.set_trace()
                    sourcefile = title.split(".")[0]
                    scan = title.split(".")[1]
                    RT_mgf=params.get('rtinseconds')
                    pepmass_mgf=params.get('pepmass')[0]
                    charge_mgf=params.get('charge')[0]
                    Spectrum = str(scan)
                intensity=list(spectrum.get('intensity array'))
                mz=list(spectrum.get('m/z array'))
                if len(intensity)!=len(mz):
                    print("alert! len(intensity)!=len(mz)",k,mgfname)
                    raise AssertionError
                mgfdata.loc[down]=[sourcefile,Spectrum,RT_mgf,pepmass_mgf,charge_mgf,intensity,mz]
                down+=1
            if mgfdata['SourceFile'][0].endswith("raw"):
                mgfdata["SourceFile"] = mgfdata['SourceFile'].str.replace(".raw","",regex=False)
            mgfdata.to_json(os.path.join(jsonfold, filename[0:-3] + "json"))
    mgf_allfiles = []
    for root, dirs, files in os.walk(jsonfold):
        for filename in files:
            mgf_allfiles.append(pd.read_json(os.path.join(jsonfold, filename)))

    mgf_allfiles = pd.concat(mgf_allfiles, ignore_index=True)
    mgf_allfiles.to_json(os.path.join(jsonfold, "SStruespectrum.json"))
    return mgf_allfiles

# ----------------------- peak picking and similarity calculation------------------------------#
#计算ppm，假设一行是一个PSM，传入的是理论文件和实际文件，具体可以再处理，可以对文件按行数得到每一个的结果
def putTintensity(toler,masses,mgfdata):
    mz = masses["FragmentMz"]  #masses.py计算得到的理论py
    ppm = 1 / 1000000
    # import ipdb
    # ipdb.set_trace()
    mz_mgf={k:v for k,v in zip(list(mgfdata["mz"][0]),list(mgfdata["intensity"][0]))}
    mzlist=sorted(mz_mgf.keys())
    mzdict={}
    # import ipdb
    # ipdb.set_trace()
    for k in mz:
        i = (np.abs(np.array(mzlist) - k)).argmin()
        # ipdb.set_trace()
        if abs(mzlist[i] - k) < k * toler * ppm:  #args.ppm=tolerance here,可以改回args版本
            mzdict[k]=mz_mgf[mzlist[i]]
        else:
            mzdict[k] = 0
    return mzdict

def putTintensity_pred(toler,masses,mgfdata):
    mz = masses["FragmentMz"]  #masses.py计算得到的理论py
    ppm = 1 / 1000000
    # import ipdb
    # ipdb.set_trace()
    mz_mgf={k:v for k,v in zip(mgfdata["mz"],mgfdata["intensity"])}
    mzlist=sorted(mz_mgf.keys())
    mzdict={}
    for k in mz:
        i = (np.abs(np.array(mzlist) - k)).argmin()
        if abs(mzlist[i] - k) < k * toler * ppm:  #args.ppm=tolerance here,可以改回args版本
            mzdict[k]=mz_mgf[mzlist[i]]
        else:
            mzdict[k] = 0
    return mzdict
# toler=20
# masses={"FragmentMz":[104.228417,123.86,113.189911]}
# putTintensity(toler,masses,mgf_allfiles)
# ----------------------- similarity calculation------------------------------#

def normalize(spectrum):
    spectrum_intensity = torch.Tensor(list(spectrum.values())) 
    spectrum_intensity = spectrum_intensity / spectrum_intensity.max()
    return spectrum_intensity

def simlarcalc(spectrum_1,spectrum_2,type): 
    #提供两种方法，开根号的cosine similarity与correlation coefficient  "cos" or "corre"
    # 开根号也可以用poisson GL代替
    spectrum_1_intensity=normalize(spectrum_1)
    spectrum_2_intensity=normalize(spectrum_2)
    # import ipdb
    # ipdb.set_trace()
    # print(spectrum_1_intensity)
    # print(spectrum_2_intensity)
    if type =="cos":
        cos = torch.nn.CosineSimilarity(dim=0)
        sim = cos(spectrum_1_intensity, spectrum_2_intensity)
    if type=="corre":
        spec=np.r_[spectrum_1_intensity.reshape(1,-1),spectrum_2_intensity.reshape(1,-1)]
        sim = np.corrcoef(spec)[0,1]
    if type=="cos_sqrt":
        cos = torch.nn.CosineSimilarity(dim=0)
        sim = cos(spectrum_1_intensity.sqrt(), spectrum_2_intensity.sqrt())
    if type=="corre_sqrt":
        spectrum_1_intensity=spectrum_1_intensity.sqrt()
        spectrum_2_intensity=spectrum_2_intensity.sqrt()
        spec=np.r_[spectrum_1_intensity.reshape(1,-1),spectrum_2_intensity.reshape(1,-1)]
        sim = np.corrcoef(spec)[0,1]
    return sim

    
# mgf_allfiles=mgf_process("/remote-home/yxwang/test/zzb/DeepGlyco/")
# print(mgf_allfiles.columns)
# spectrum_1={1:10,2:20,4:30}
# spectrum_2={1:10,2:20,3:10,5:20}
# print(spectrum_1.keys() ^ spectrum_2.keys())
# for i in spectrum_1.keys() ^ spectrum_2.keys():
#     if i not in spectrum_1.keys():
#         spectrum_1[i]=0
#     if i not in spectrum_2.keys():
#         spectrum_2[i]=0
# spectrum_1=dict(sorted(spectrum_1.items(), key=lambda x: x[0]))
# spectrum_2=dict(sorted(spectrum_2.items(), key=lambda x: x[0]))
# print(spectrum_1)
# print(spectrum_2)
# sim=simlarcalc(spectrum_1,spectrum_2,"cos")  #感觉cos好一点，不知道是不是我用的corre有问题，感觉对于变化不敏感
# print("median",np.median(scorelist))
