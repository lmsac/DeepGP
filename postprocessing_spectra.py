#对预测得到的结果转化为谱图的形式
#目前改了by和BY，exp暂时还没改
import pandas as pd
import numpy as np
import ipdb
from utils import *
import torch
import masses
from masses import *
# ----------------------- dict ------------------------------#
MAXFICHARGE = int(num_col/12)
if MAXFICHARGE == 2:
    ionname_by = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
        "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh".split(
        ',')
    ionname_BY = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
                "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f".split(
                ',')

if MAXFICHARGE == 3:
    ionname_by = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
        "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh,"\
        "b3,b3n,b3o,b3h,b3nh,b3oh,y3,y3n,y3o,y3h,y3nh,y3oh".split(
        ',')
    ionname_BY = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
        "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f,"\
        "B3,B3n,B3o,B3f,Y3,Y3n,Y3o,Y3f".split(
        ',')
lossdict_by = {"noloss": '', "H2O": 'o', "NH3": 'n',  '(HexNAc)1': 'h',
                '1(NH3)1(+HexNAc)': 'nh', '1(H2O)1(+HexNAc)': 'oh'}
lossdict_BY = {"noloss": '', "H2O": 'o', "NH3": 'n', 'FUC': 'f'}
fields_byBY = "SourceFile,id,PEP.StrippedSequence,iden_pep,PlausibleStruct,PP.Charge,metric,FragmentMz,FI.FrgType,FI.LossType,FI.Charge,FI.Intensity,FI.FrgNum,Precurmass_cal".split(',')
fields_exp = "SourceFile,id,PEP.StrippedSequence,iden_pep,PlausibleStruct,PP.Charge,FragmentMz,FI.FrgType,FI.LossType,FI.Charge,FI.Intensity,FI.FrgNum,Precurmass_cal".split(',')
    
# ----------------------- func ------------------------------#

def byBYextract(row):
    outputdataframe1 = pd.DataFrame(columns=fields_byBY)
    # StrippedSequence = "".join([vocab.to_word(i) for i in eval(row['repsequence'])])
    StrippedSequence=row["ipeptide"]
    PPCharge=int(row["charge"])
    metric=float(row["metric"])
    id=row["id"]
    global id_glyspec
    global id_iden_pep
    glyspec=id_glyspec[id]
    ms2by=row["ms2by"]
    ms2BY=row["ms2BY"]
    iden_pep=id_iden_pep[id]
    

    # import ipdb
    # ipdb.set_trace()
    PlausibleStruct=iden_pep.split("_")[-1]
    glysite=iden_pep.split("_")[2]
    # Precurmass_exp=glyspec_pepmass_mgf[glyspec]
    Precurmass_cal= (calcModpepMass(peptide_process(iden_pep))+MASS["H2O"])/PPCharge+MASS["H+"]#根据iden_pep计算母离子质荷比
    # deltamz=abs(Precurmass_exp-Precurmass_cal)/Precurmass_exp
    FRAG_MODE=["HCD_by","HCD_1"]
    mz_calc=masses.pepfragmass(iden_pep,FRAG_MODE) #得到iontype和m/z对
    # ipdb.set_trace()  
    #by  
    jmax=ms2by.size()[1]
    imax=ms2by.size()[0]
    # ipdb.set_trace()
    assert jmax==num_col
    for j in range(0,jmax):
        for i in range(0,imax):
            intensity_pred=ms2by[i][j] #从ms2中提取对应位置的强度
            if intensity_pred <=0:
                continue
            intensity_pred=float(intensity_pred)
            ionname=ionname_by[j] #从ionname列表中获得对应结果的离子类型
            FIFrgType=ionname[0]
            FICharge=int(ionname[1])
            FILossType=ionname[2:]
            FILossType=list(lossdict_by.keys())[list(lossdict_by.values()).index(FILossType)]
            assert FIFrgType in ["b","y"]
            # ipdb.set_trace()
            if FIFrgType=="b":
                FragmentNumber=i+1
                masskey="b" + str(FragmentNumber) + "_" + str(FICharge)
            if FIFrgType=="y":
                FragmentNumber=imax - i
                masskey="y" + str(FragmentNumber) + "_" + str(FICharge)
            # import ipdb
            if FIFrgType=="b" and int(FragmentNumber)<=int(glysite) and ionname.endswith("h"):
                continue
            if FIFrgType=="y" and int(i+1)>int(glysite) and ionname.endswith("h"):
                continue
            if FILossType!="noloss":
                loss=ionname[2]
                if loss !="h":
                    loss=list(lossdict_by.keys())[list(lossdict_by.values()).index(loss)]
                    masskey+="_loss_"+FILossType
            if ionname.endswith("h"):
                masskey+="_"+"(HexNAc)1"
                # import ipdb
                # ipdb.set_trace()
            else:
                masskey+="_"+"(HexNAc)0"

            counter=0
            for k in mz_calc: 
                for dict in k:
                    key=[str(key) for key in dict.keys()]
                    assert len(key)==1
                    if masskey == key[0]:
                        value=[float(value) for value in dict.values()]
                        assert len(value)==1
                        FragmentMz=value[0]
                        counter+=1
                        if counter>1:
                            raise ValueError("loop number more than one")
            # ipdb.set_trace()
            if counter==1:
                outputdataframe1.loc[len(outputdataframe1)] = [glyspec,id, StrippedSequence,iden_pep,PlausibleStruct,\
                                                           PPCharge,metric,FragmentMz,\
                                                           FIFrgType, FILossType,FICharge,intensity_pred,\
                                                           FragmentNumber,Precurmass_cal]                       


    #BY
    jmax=ms2BY.size()[1]
    imax=ms2BY.size()[0]
    assert jmax==num_col*2/3
    for j in range(0,jmax):
        for i in range(0,imax):
            intensity_pred=ms2BY[i][j] #从ms2中提取对应位置的强度
            if intensity_pred <=0:
                continue
            intensity_pred=float(intensity_pred)
            ionname=ionname_BY[j] #从ionname列表中获得对应结果的离子类型
            FIFrgType=ionname[0]
            FICharge=int(ionname[1])
            FILossType=ionname[2:]
            if FILossType=="f" and "F" not in PlausibleStruct:
                continue
            if FILossType=="f" and FIFrgType=="B":
                continue
            
            FILossType=list(lossdict_BY.keys())[list(lossdict_BY.values()).index(FILossType)]
            FragmentNumber=i
            
            #"_" + str(k) + "_" + str(ficharge)+"_loss_H2O"
            masskey="_" + str(FragmentNumber) + "_" + str(FICharge)
            if FILossType!="noloss":
                masskey+="_loss_"+FILossType
            counter=0
            # if FILossType=="FUC":
            #     ipdb.set_trace()
            for k in mz_calc: 
                for dict in k:
                    key=[str(key) for key in dict.keys()]
                    assert len(key)==1
                    if FIFrgType == key[0][0]:
                        if key[0].endswith(masskey):
                            value=[float(value) for value in dict.values()]
                            assert len(value)==1
                            FragmentMz=value[0]
                            counter+=1
                            if counter>1:
                                raise ValueError("loop number more than one")
            # import ipdb
            # ipdb.set_trace()
            if counter==1:
                outputdataframe1.loc[len(outputdataframe1)] = [glyspec,id, StrippedSequence,iden_pep,PlausibleStruct,\
                                                           PPCharge,metric,FragmentMz,\
                                                           FIFrgType, FILossType,FICharge,intensity_pred,\
                                                           FragmentNumber,Precurmass_cal]
            
    # import ipdb
    # ipdb.set_trace()
    return outputdataframe1

def expbyBYextract(row):
    outputdataframe1 = pd.DataFrame(columns=fields_exp)
    StrippedSequence = row["peptide"]
    PPCharge=int(row["charge"])
    id=row["_id"]
    glyspec=row["GlySpec"]
    # import ipdb
    # ipdb.set_trace()
    ms2by=row["ions_by"]
    ms2BY=row["ions_BY"]
    # iden_pep=id_iden_pep[id]
    # import ipdb
    # ipdb.set_trace()
    iden_pep=row["iden_pep"]
    PlausibleStruct=iden_pep.split("_")[-1]
    glysite=iden_pep.split("_")[2]
    # Precurmass_exp=glyspec_pepmass_mgf[glyspec]
    Precurmass_cal= (calcModpepMass(peptide_process(iden_pep))+MASS["H2O"])/PPCharge+MASS["H+"]#根据iden_pep计算母离子质荷比
    # deltamz=abs(Precurmass_exp-Precurmass_cal)/Precurmass_exp
    FRAG_MODE=["HCD_by","HCD_1"]
    mz_calc=masses.pepfragmass(iden_pep,FRAG_MODE) #得到iontype和m/z对
    # ipdb.set_trace()  
    #by  
    jmax=ms2by.size()[1]
    imax=ms2by.size()[0]
    # ipdb.set_trace()
    assert jmax==num_col
    for j in range(0,jmax):
        for i in range(0,imax):
            intensity_pred=ms2by[i][j] #从ms2中提取对应位置的强度
            if intensity_pred <=0:
                continue
            intensity_pred=float(intensity_pred)
            ionname=ionname_by[j] #从ionname列表中获得对应结果的离子类型
            FIFrgType=ionname[0]
            FICharge=int(ionname[1])
            FILossType=ionname[2:]
            FILossType=list(lossdict_by.keys())[list(lossdict_by.values()).index(FILossType)]
            assert FIFrgType in ["b","y"]
            # ipdb.set_trace()
            if FIFrgType=="b":
                FragmentNumber=i+1
                masskey="b" + str(FragmentNumber) + "_" + str(FICharge)
            if FIFrgType=="y":
                FragmentNumber=imax - i
                masskey="y" + str(FragmentNumber) + "_" + str(FICharge)
            # import ipdb
            if FIFrgType=="b" and int(FragmentNumber)<=int(glysite) and ionname.endswith("h"):
                continue
            if FIFrgType=="y" and int(i+1)>int(glysite) and ionname.endswith("h"):
                continue
            if FILossType!="noloss":
                loss=ionname[2]
                if loss !="h":
                    loss=list(lossdict_by.keys())[list(lossdict_by.values()).index(loss)]
                    masskey+="_loss_"+FILossType
            if ionname.endswith("h"):
                masskey+="_"+"(HexNAc)1"
                # import ipdb
                # ipdb.set_trace()
            else:
                masskey+="_"+"(HexNAc)0"

            counter=0
            for k in mz_calc: 
                for dict in k:
                    key=[str(key) for key in dict.keys()]
                    assert len(key)==1
                    if masskey == key[0]:
                        value=[float(value) for value in dict.values()]
                        assert len(value)==1
                        FragmentMz=value[0]
                        counter+=1
                        if counter>1:
                            raise ValueError("loop number more than one")
            # ipdb.set_trace()
            if counter==1:
                outputdataframe1.loc[len(outputdataframe1)] = [glyspec,id, StrippedSequence,iden_pep,PlausibleStruct,\
                                                           PPCharge,FragmentMz,\
                                                           FIFrgType, FILossType,FICharge,intensity_pred,\
                                                           FragmentNumber,Precurmass_cal]                       


    #BY
    jmax=ms2BY.size()[1]
    imax=ms2BY.size()[0]
    assert jmax==num_col*2/3
    for j in range(0,jmax):
        for i in range(0,imax):
            intensity_pred=ms2BY[i][j] #从ms2中提取对应位置的强度
            if intensity_pred <=0:
                continue
            intensity_pred=float(intensity_pred)
            ionname=ionname_BY[j] #从ionname列表中获得对应结果的离子类型
            FIFrgType=ionname[0]
            FICharge=int(ionname[1])
            FILossType=ionname[2:]
            if FILossType=="f" and "F" not in PlausibleStruct:
                continue
            if FILossType=="f" and FIFrgType=="B":
                continue
            
            FILossType=list(lossdict_BY.keys())[list(lossdict_BY.values()).index(FILossType)]
            FragmentNumber=i
            
            #"_" + str(k) + "_" + str(ficharge)+"_loss_H2O"
            masskey="_" + str(FragmentNumber) + "_" + str(FICharge)
            if FILossType!="noloss":
                masskey+="_loss_"+FILossType
            counter=0
            # if FILossType=="FUC":
            #     ipdb.set_trace()
            for k in mz_calc: 
                for dict in k:
                    key=[str(key) for key in dict.keys()]
                    assert len(key)==1
                    if FIFrgType == key[0][0]:
                        if key[0].endswith(masskey):
                            value=[float(value) for value in dict.values()]
                            assert len(value)==1
                            FragmentMz=value[0]
                            counter+=1
                            if counter>1:
                                raise ValueError("loop number more than one")
            # import ipdb
            # ipdb.set_trace()
            if counter==1:
                outputdataframe1.loc[len(outputdataframe1)] = [glyspec,id, StrippedSequence,iden_pep,PlausibleStruct,\
                                                           PPCharge,FragmentMz,\
                                                           FIFrgType, FILossType,FICharge,intensity_pred,\
                                                           FragmentNumber,Precurmass_cal]
            
    # import ipdb
    # ipdb.set_trace()
    return outputdataframe1

def byBYextract_reidentification(row):
    outputdataframe1 = pd.DataFrame(columns=["GlySpec","modelpep","FragmentMz",\
                                             "FIFrgType", "FILossType","FICharge","intensity_pred",\
                                             "FragmentNumber","FragmentGlycan"])
    import ipdb
    GlySpec=row["GlySpec"]
    FRAG_MODE=["HCD_1"]
    # ipdb.set_trace()  

    #BY
    for num in ["1","2"]:
        modelpep=row["modelpep"+num]
        ms2BY=row["maskmodel"+num]
        mz_calc=masses.pepfragmass(modelpep,FRAG_MODE) #得到iontype和m/z对
        # ipdb.set_trace()
        jmax=ms2BY.size()[1]
        imax=ms2BY.size()[0]
        assert jmax==num_col*2/3
        # ipdb.set_trace()
        for j in range(0,jmax):
            for i in range(0,imax):
                intensity_pred=ms2BY[i][j] #从ms2中提取对应位置的强度
                if intensity_pred <=0:
                    continue
                intensity_pred=float(intensity_pred)
                ionname=ionname_BY[j] #从ionname列表中获得对应结果的离子类型
                FIFrgType=ionname[0]
                FICharge=int(ionname[1])
                FILossType=ionname[2:]
                
                FILossType=list(lossdict_BY.keys())[list(lossdict_BY.values()).index(FILossType)]
                FragmentNumber=i
                masskey="_" + str(FragmentNumber) + "_" + str(FICharge)
                if FILossType!="noloss":
                    masskey+="_loss_"+FILossType
                counter=0
                # ipdb.set_trace()
                for k in mz_calc: 
                    for dict in k:
                        key=[str(key) for key in dict.keys()]
                        assert len(key)==1
                        if FIFrgType == key[0][0]:
                            if key[0].endswith(masskey):
                                value=[float(value) for value in dict.values()]
                                assert len(value)==1
                                FragmentMz=value[0]
                                fragnode=str(key[0].split("_")[0][1:])
                                # ipdb.set_trace()
                                counter+=1
                                if counter>1:
                                    raise ValueError("loop number more than one")
                # import ipdb
                # ipdb.set_trace()
                if counter==1:
                    outputdataframe1.loc[len(outputdataframe1)] = [GlySpec,modelpep,FragmentMz,\
                                                            FIFrgType, FILossType,FICharge,intensity_pred,\
                                                            FragmentNumber,fragnode]
            
    # import ipdb
    # ipdb.set_trace()
    return outputdataframe1
# ----------------------- execute ------------------------------#
def postprocessing(savename):
    filename=savename+"_byBY_result.csv"
    df=pd.read_csv(filename)
    # import ipdb
    # ipdb.set_trace()
    # df=df.loc[df["metric"]==max(df["metric"])] #取一个进行测试
    # df.to_csv(datafold+"del.csv",index=False)
    print(df.columns)
    df["ms2by"]=df["ms2by"].apply(lambda x:torch.Tensor(eval(x)))
    df["ms2BY"]=df["ms2BY"].apply(lambda x:torch.Tensor(eval(x)))
    #训练的时候ms2使用了log2（x+1)，这里变回来#检查一下
    #'repsequence', 'charge', 'ipeptide', 'iPlausibleStruct', 'ms2', 'cos','id'
    print(f"The number of the lines of the input files {len(df)}")
    filename_expjson=savename+"_byBY_testdata.json"
    dfexp=pd.read_json(filename_expjson)
    # dfexp=dfexp.loc[dfexp["_id"]==2665] 
    # dfexp.to_csv(datafold+"exp_6923_del.csv",index=False)
    print(dfexp.columns)
    # import ipdb
    # ipdb.set_trace()
    global id_glyspec
    global id_iden_pep
    id_glyspec = dict(zip(dfexp['_id'],dfexp['GlySpec'])) #产生_id,GlySpec字典
    id_iden_pep=dict(zip(dfexp['_id'],dfexp['iden_pep']))
    # import ipdb
    # ipdb.set_trace()
    # dfinput=pd.read_csv(dfinput)
    # ipdb.set_trace()
    outputdataframe_BY=pd.concat(df.apply(byBYextract,axis=1).to_list())
    outputdataframe_BY.reset_index(inplace=True,drop=True)
    print(f"The number of the lines of the output files {len(outputdataframe_BY)}")
    outputdataframe_BY.to_csv(savename+"_predictmonomz.csv",index=False)


    
    # import ipdb
    # ipdb.set_trace()
    dfexp["ions_by"]=dfexp["ions_by"].apply(lambda x:torch.Tensor(x))
    dfexp["ions_BY"]=dfexp["ions_BY"].apply(lambda x:torch.Tensor(x))
    print(dfexp.columns)
    # dfexp.to_csv(datafold+"exp_6923_del.csv",index=False
    print(f"The number of the lines of the input files {len(dfexp)}")
    # outputdataframe_expBY = pd.DataFrame(columns=fields_exp)
    # import ipdb
    # ipdb.set_trace()
    outputdataframe_expBY=pd.concat(dfexp.apply(expbyBYextract,axis=1).to_list())
    outputdataframe_expBY.reset_index(inplace=True,drop=True)
    print(f"The number of the lines of the output files {len(outputdataframe_expBY)}")
    outputdataframe_expBY.to_csv(savename+"_expmonomz.csv",index=False)

def Precurmass_cal(innstance):
    iden_pep=innstance["iden_pep"]
    PPCharge=int(innstance["charge"])
    Precurmass_cal=(calcModpepMass(peptide_process(iden_pep))+MASS["H2O"])/PPCharge+MASS["H+"]
    return Precurmass_cal
def postprocessing_nointensity(savename,dfinputname):
    filename=savename+"_byBY_result.csv"
    df=pd.read_csv(filename)
    print(df.columns)
    print(f"The number of the lines of the input files {len(df)}")
    filename_expjson=savename+"_byBY_testdata.json"
    dfexp=pd.read_json(filename_expjson)
    print(dfexp.columns)
    print(f"The number of the lines of the input exp files {len(dfexp)}")
    # import ipdb
    # ipdb.set_trace()
    dfexp=dfexp[['_id','GlySpec',"iden_pep"]]
    df=df[[ 'charge', 'ipeptide', 'iPlausibleStruct','metric', "metricBY_cos",'id']]
    df.columns=['charge', 'PEP.StrippedSequence', 'PlausibleStruct','metric',"metricBY", '_id']
    df=pd.merge(df,dfexp,on="_id",how="left")
    # import ipdb
    # ipdb.set_trace()
    dfinput=pd.read_csv(dfinputname)
    # ipdb.set_trace()
    dfinput=dfinput[['GlySpec','pepmass_mgf',"matching_ration","iden_pep"]].drop_duplicates()
    print(len(dfinput))
    df=pd.merge(df,dfinput,on=["GlySpec","iden_pep"],how="left")
    # ipdb.set_trace()
    df["Precurmass_cal"]= df.apply(Precurmass_cal,axis=1)
    df["deltamzpercent"]=abs(df["pepmass_mgf"]-df["Precurmass_cal"])/df["Precurmass_cal"]
    df.to_csv(savename+"_nointensity_predictmonomz.csv",index=False)


if __name__ == '__main__':  

    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/IgG/PXD015360_IgG/test_replace_predict/test_byBY_PXD015360_IgG_data_1st_target_decoy_model"
    # dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/IgG/PXD015360_IgG/PXD015360_IgG_data_1st_redo1_nocoelute_target_decoy_Retained_all.csv"
    # postprocessing_nointensity(savename,dfinputname)


    savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD009654/test_replace_predict/test_byBY_PXD009654_human_data_1st_target_decoy_model"
    dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD009654/PXD009654_data_1st_redo1_nocoelute_target_decoy_Retained_all.csv"
    postprocessing_nointensity(savename,dfinputname)

    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/IgG/PXD009716/test_replace_predict/PXD009716_IgG_data_1st_target_decoy_NAGF_model_0.930816"
    # dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/IgG/PXD009716/PXD009716_IgG_data_1st_target_decoy_Retained_all.csv"
    # postprocessing_nointensity(savename,dfinputname)


    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/syn/PXD023980/test_replace_predict/test_byBY_PXD023980_syn_test_9_glycans_redo1_model_0.927153"
    # dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/syn/PXD023980/PXD023980_Syn_test_9_glycans_redo1_data_1st.csv"
    # postprocessing_nointensity(savename,dfinputname)
    
    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD016428/test_replace_predict/test_byBY_PXD016428_IgG_data_1st_target_decoy_model_0.933599"
    # dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD016428/PXD016428_serum_data_1st_target_decoy_Retained_all.csv"
    # postprocessing_nointensity(savename,dfinputname)


    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD016428/test_replace_predict/test_byBY_PXD016428_IgG_data_1st_target_decoy_model_0.937599"
    # dfinputname="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/PXD016428/PXD016428_serum_data_1st_target_decoy_Retained_all.csv"
    # postprocessing_nointensity(savename,dfinputname)

    # savename="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/Five_tissues/test_replace_predict/test_byBY_PXD005411_model_0.927740"
    # postprocessing(savename)