import numpy as np
import pandas as pd
import os
import os,sys
import sys
sys.path.append('..')
import masses
from utils import *
MAXFICHARGE = int(num_col/12)
# ---------------------- decoration processing------------------------#
def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    glyco = []
    number = "0123456789"
    l = len(presequence)
    sig=0
    for i in range(l):
        if presequence[i]=='4':
            assert i==0
            sig=1
            glyco.append(4)
            continue
        elif presequence[i] in number:
            glyco[-1] = int(presequence[i])
            continue
        glyco.append(0)
        if sig:
            sig=0
            glyco.pop()
    return glyco

# ----------------------- by prediction ------------------------------#
def resultreport2matrix(filename,outputfile):
    fields = "GlySpec,sequence,decoration,decoration_ACE,PlausibleStruct,charge,ions,iden_pep".split(',')
    outputdataframe = pd.DataFrame(columns=fields)
    testdata = pd.read_csv(filename)
    decorationdict={}
    decorationdict['PHO']=1
    decorationdict['GLY']=5
    #糖没有代替原来phos的位置，而是随机初始化了，希望可以同时处理糖和磷酸化，做后修饰 crosstalk
    #糖基化后续可以随机初始化或者复制磷酸化的赋值
    decorationdict['OXI']=2
    decorationdict['CAR'] = 3
    decorationdict['ACE'] = 4
    #这里是糖基化是代替了phos的一号位，也可以随机初始化，放到五号位
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n',  '(HexNAc)1': 'h',"(HexNAc)0":"",
                '1(NH3)1(+HexNAc)': 'nh', '1(H2O)1(+HexNAc)': 'oh'}
    #改：HexNac增加作为n，如果没有糖基位点，那么不可以增加
    testdata=testdata[['GlySpec', 'Charge', 'Peptide', 'Mod', 'PlausibleStruct',
       'GlySite', 'iden_pep', 'ions']]
    testdata.columns=['GlySpec', 'Charge', 'Peptide', 'Mod', 'PlausibleStruct',
       'GlySite', 'iden_pep', 'ions']
    down = 0
    for index,row in testdata.iterrows(): #如果有重复的iden_pep这就会产生问题，所以换成一行一行处理
        GlySpec=row["GlySpec"]
        iden_pep=row["iden_pep"]
        iden_pep_list=masses.peptide_process(iden_pep)
        # print(iden_pep_list)
        sequence=iden_pep_list["sequence"].replace("J","N")
        decoration=iden_pep_list["modifications"]
        length=len(sequence)
        decoration_intial=[0]*length
        decoration_ACE=[0]*length  #把4单独存一个list，这样可以处理首位有两个修饰的情况
        for k in decoration:
            if k["type"]=="mod":
                name=k["name"].lstrip("(")
                name=name.rstrip(")1").upper()
                if name=="ACE":
                    decoration_ACE[0]=decorationdict[name]
                else:
                    site=k["position"]
                    decoration_intial[site]=decorationdict[name]
            if k["type"]=="glyco":
                glysite=k["position"]
                PlausibleStruct=k["structure"]
                decoration_intial[glysite]=decorationdict["GLY"] #将glysite特意存了下来作为hexnac丢失是否应该存在的标志
        # print("decoration_intial",decoration_intial)
        charge = int(iden_pep_list["charge"])
        #ions
        if MAXFICHARGE == 2:
            ionname = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
                "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh".split(
                ',')
        if MAXFICHARGE == 3:
            ionname = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
                "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh,"\
                "b3,b3n,b3o,b3h,b3nh,b3oh,y3,y3n,y3o,y3h,y3nh,y3oh".split(
                ',')
        ions = np.zeros((length - 1, len(ionname)))
        # print(ions)
        # print(ionname)
        # print(len(ionname))
        # ipdb.set_trace()
        # print(frame)
        ion = eval(row['ions'])
        # ipdb.set_trace()
        # print("ion",ion)
        m=ion.values()
        # print("m",m)
        for ion_1 in m:
            # print("ion_1",ion_1)
            ion_type=ion_1[0]
            # print("ion_type_ini",ion_type)
            # ipdb.set_trace()
            ion_type_by=[]
            for ion_type1 in ion_type:
                assert ion_type1[0].lower()=="y" or ion_type1[0].lower()=="b"
                ficharge=int(ion_type1.split("_")[1][0])
                if ion_type1[0].islower():
                    if  ficharge<=MAXFICHARGE:
                        by=ion_type1[0]
                        if by=="b" and int(ion_type1.split("_")[0][1:])<=int(glysite) and "(HexNAc)1" in ion_type1:
                            pass
                        elif by=="y" and (length-int(ion_type1.split("_")[0][1:]))>int(glysite) and "(HexNAc)1" in ion_type1:
                            pass
                        else:
                            ion_type_by.append(ion_type1)
            # print(iden_pep)
            # print("ion_type_by",ion_type_by)
            # print("ion_type_by",ion_type_by)
            # import ipdb
            # ipdb.set_trace()
            ion_type_len=len(ion_type_by)
            if ion_type_len==0:
                continue
            else:
                intensity=ion_1[1]/ion_type_len
                # print("intensity",intensity)
                for k in ion_type_by:
                    FragmentType=k[0]
                    FragmentNumber=int(k.split("_")[0][1:])
                    Fragmentcharge=int(k.split("_")[1])
                    if Fragmentcharge > 3:
                        continue
                    if "loss" in k:
                        Fragmentloss=k.split("_")[-2].upper()
                        Fragmentloss=Fragmentloss.replace("H20","H2O")
                        Fragmentloss=lossdict[Fragmentloss]
                    else:
                        Fragmentloss=""
                    # print("Fragmentloss",Fragmentloss)
                    Fragmenthexnac=k.split("_")[-1]
                    Fragmenthexnac=lossdict[Fragmenthexnac]
                    # print("Fragmenthexnac",Fragmenthexnac)
                    ion_k=FragmentType+str(Fragmentcharge)+Fragmentloss+Fragmenthexnac
                    # print("ion_k",ion_k)
                    # print("ionname",ionname)
                    j = ionname.index(ion_k)
                    # print("j",j)
                    i = FragmentNumber - 1 if FragmentType == 'b' else length - FragmentNumber - 1
                    # print(i,j)
                    ions[i, j] = intensity
        if not ions.any():
            print(GlySpec)
            import ipdb
            ipdb.set_trace()
            #为什么ions全为0就会变成空值
        else:
            outputdataframe.loc[down] = [GlySpec,sequence, decoration_intial,decoration_ACE,PlausibleStruct,charge, ions,iden_pep]
            down += 1
    outputdataframe.to_json(outputfile)
    print("finish")
    print(outputdataframe)


# resultreport2matrix("/remote-home/yxwang/test/zzb/DeepGlyco/20221113_pglyco_traincsv_test.csv")
# ----------------------- BY prediction ------------------------------#
def report2BYmatrix(filename,outputfile):
    fields = "GlySpec,sequence,decoration,decoration_ACE,PlausibleStruct,charge,ions,iden_pep".split(',')
    outputdataframe = pd.DataFrame(columns=fields)
    testdata = pd.read_csv(filename)
    decorationdict={}
    decorationdict['PHO']=1
    decorationdict['GLY']=5
    #糖没有代替原来phos的位置，而是随机初始化了，希望可以同时处理糖和磷酸化，做后修饰 crosstalk
    #糖基化后续可以随机初始化或者复制磷酸化的赋值
    decorationdict['OXI']=2
    decorationdict['CAR'] = 3
    decorationdict['ACE'] = 4
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', 'FUC': 'f'}
    #这里是将HexNAc作为增加，这样就可以不用考虑糖基位点，默认为没有糖
    # print("testdata",testdata)
    # print(testdata.columns)
    testdata=testdata[['GlySpec', 'Charge', 'Peptide', 'Mod', 'PlausibleStruct',
       'GlySite', 'iden_pep', 'ions']]
    down = 0
    for index,row in testdata.iterrows():
        GlySpec=row["GlySpec"]
        iden_pep=row["iden_pep"]
        iden_pep_list=masses.peptide_process(iden_pep)
        # ipdb.set_trace()
        # print(iden_pep_list)
        sequence=iden_pep_list["sequence"].replace("J","N")
        decoration=iden_pep_list["modifications"]
        length=len(sequence)
        decoration_intial=[0]*length
        decoration_ACE=[0]*length  #把4单独存一个list，这样可以处理首位有两个修饰的情况
        for k in decoration:
            if k["type"]=="mod":
                # ipdb.set_trace()
                name=k["name"].lstrip("(")
                name=name.rstrip(")1").upper()
                if name=="ACE":
                    decoration_ACE[0]=decorationdict[name]
                else:
                    site=k["position"]
                    decoration_intial[site]=decorationdict[name]
            if k["type"]=="glyco":
                site=k["position"]
                PlausibleStruct=k["structure"]
                decoration_intial[site]=decorationdict["GLY"]
        # print("decoration_intial",decoration_intial)
        charge = int(iden_pep_list["charge"])
        
        #ions
        if MAXFICHARGE == 2:
            ionname = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
                "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f".split(
                ',') #也可以设置多丢失，例如B1oh
        if MAXFICHARGE == 3:
            ionname = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
                "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f,"\
                "B3,B3n,B3o,B3f,Y3,Y3n,Y3o,Y3f".split(
                ',')
        length_glyco=PlausibleStruct.count("(")
        ions = np.zeros((length_glyco, len(ionname))) #是糖的边的数目，而不再是肽段的数目
        #这个矩阵具有轮换性，跟边的index排序没什么关系
        # print(ions)
        # print(ionname)
        # print(len(ionname))
        # ipdb.set_trace()
        # print(frame)
        ion = eval(row['ions'])
        # print("ion",ion)
        m=ion.values()
        # print("m",m)
        for ion_1 in m:
            # print("ion_1",ion_1)
            ion_type=ion_1[0]
            # print("ion_type_ini",ion_type)
            # ipdb.set_trace()
            ion_type_BY=[]
            for ion_type1 in ion_type:
                assert ion_type1[0].lower()=="y" or ion_type1[0].lower()=="b"
                # print(ion_type1)
                # ipdb.set_trace()
                if ion_type1[0].isupper():
                    ficharge=int(ion_type1.split("_")[2])
                    if  ficharge<=MAXFICHARGE:
                        ion_type_BY.append(ion_type1)
            ion_type_len=len(ion_type_BY)
            if ion_type_len==0:
                continue
            else:
                intensity=ion_1[1]/ion_type_len
                # print("intensity",intensity)
                for k in ion_type_BY:
                    FragmentType=k[0]
                    FragmentNumber=int(k.split("_")[1])
                    Fragmentcharge=int(k.split("_")[2])
                    if Fragmentcharge > 3:
                        continue
                    if "loss" in k:
                        Fragmentloss=k.split("_")[-1].upper()
                        Fragmentloss=Fragmentloss.replace("H20","H2O")
                        Fragmentloss=lossdict[Fragmentloss]
                    else:
                        Fragmentloss=""
                    # print("Fragmentloss",Fragmentloss)
                    # print("Fragmenthexnac",Fragmenthexnac)
                    ion_k=FragmentType+str(Fragmentcharge)+Fragmentloss
                    # print("ion_k",ion_k)
                    # print("ionname",ionname)
                    j = ionname.index(ion_k)
                    # print("j",j)
                    i = FragmentNumber
                    # print(i,j)
                    # print(k)
                    ions[i, j] = intensity
        if not ions.any():
            print(GlySpec)
        else:
            outputdataframe.loc[down] = [GlySpec,sequence, decoration_intial,decoration_ACE,PlausibleStruct,charge, ions,iden_pep]
            down += 1
    outputdataframe.to_json(outputfile)
    print("finish")
    print(outputdataframe)

# report2BYmatrix("/remote-home/yxwang/test/zzb/DeepGlyco/20230222_pglyco_traincsv_test_BY.csv")

# ----------------------- byBY prediction ------------------------------#
#融合by和BY的预处理
def byBYmatrix(filename,outputfile):
    fields = "GlySpec,sequence,decoration,decoration_ACE,PlausibleStruct,charge,ions_by,ions_BY,iden_pep".split(',')
    outputdataframe = pd.DataFrame(columns=fields)
    testdata = pd.read_csv(filename)
    decorationdict={}
    decorationdict['PHO']=1
    decorationdict['GLY']=5
    #糖没有代替原来phos的位置，而是随机初始化了，希望可以同时处理糖和磷酸化，做后修饰 crosstalk
    decorationdict['OXI']=2
    decorationdict['CAR'] = 3
    decorationdict['ACE'] = 4
    lossdict_by = {"noloss": '', "H2O": 'o', "NH3": 'n',  '(HexNAc)1': 'h',"(HexNAc)0":"",
                '1(NH3)1(+HexNAc)': 'nh', '1(H2O)1(+HexNAc)': 'oh'}
    lossdict_BY = {"noloss": '', "H2O": 'o', "NH3": 'n', 'FUC': 'f'}
    #改：HexNac增加作为n，如果没有糖基位点，那么不可以增加
    testdata=testdata[['GlySpec', 'Charge', 'Peptide', 'Mod', 'PlausibleStruct',
       'GlySite', 'iden_pep', 'ions']]
    for index,row in testdata.iterrows(): #如果有重复的iden_pep这就会产生问题，所以换成一行一行处理
        GlySpec=row["GlySpec"]
        iden_pep=row["iden_pep"]
        iden_pep_list=masses.peptide_process(iden_pep)
        # print(iden_pep_list)
        sequence=iden_pep_list["sequence"].replace("J","N")
        decoration=iden_pep_list["modifications"]
        length=len(sequence)
        decoration_intial=[0]*length
        decoration_ACE=[0]*length  #把4单独存一个list，这样可以处理首位有两个修饰的情况
        for k in decoration:
            if k["type"]=="mod":
                name=k["name"].lstrip("(")
                name=name.rstrip(")1").upper()
                if name=="ACE":
                    decoration_ACE[0]=decorationdict[name]
                else:
                    site=k["position"]
                    decoration_intial[site]=decorationdict[name]
            if k["type"]=="glyco":
                glysite=k["position"]
                PlausibleStruct=k["structure"]
                decoration_intial[glysite]=decorationdict["GLY"] #将glysite特意存了下来作为hexnac丢失是否应该存在的标志
        # print("decoration_intial",decoration_intial)
        charge = int(iden_pep_list["charge"])
        #ions
        if MAXFICHARGE == 2:
            ionname_by = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
                "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh".split(
                ',')
            ionname_BY = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
                "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f".split(
                ',') #也可以设置多丢失，例如B1oh
        if MAXFICHARGE == 3:
            ionname_by = "b1,b1n,b1o,b1h,b1nh,b1oh,y1,y1n,y1o,y1h,y1nh,y1oh,"\
                "b2,b2n,b2o,b2h,b2nh,b2oh,y2,y2n,y2o,y2h,y2nh,y2oh,"\
                "b3,b3n,b3o,b3h,b3nh,b3oh,y3,y3n,y3o,y3h,y3nh,y3oh".split(
                ',')
            ionname_BY = "B1,B1n,B1o,B1f,Y1,Y1n,Y1o,Y1f,"\
                "B2,B2n,B2o,B2f,Y2,Y2n,Y2o,Y2f,"\
                "B3,B3n,B3o,B3f,Y3,Y3n,Y3o,Y3f".split(
                ',')
        ions_by = np.zeros((length - 1, len(ionname_by)))
        length_glyco=PlausibleStruct.count("(")
        ions_BY = np.zeros((length_glyco, len(ionname_BY)))
        ion = eval(row['ions'])
        m=ion.values()
        # print("m",m)
        for ion_1 in m:
            # print("ion_1",ion_1)
            ion_type=ion_1[0]
            # print("ion_type_ini",ion_type)
            # ipdb.set_trace()
            ion_type_by=[]
            ion_type_BY=[]
            # import ipdb
            # ipdb.set_trace()
            for ion_type1 in ion_type:
                assert ion_type1[0].lower()=="y" or ion_type1[0].lower()=="b"
                if ion_type1[0].islower():
                    ficharge=int(ion_type1.split("_")[1][0])
                    if  ficharge<=MAXFICHARGE:
                        by=ion_type1[0]
                        if by=="b" and int(ion_type1.split("_")[0][1:])<=int(glysite) and "(HexNAc)1" in ion_type1:
                            pass
                        elif by=="y" and (length-int(ion_type1.split("_")[0][1:]))>int(glysite) and "(HexNAc)1" in ion_type1:
                            pass
                        else:
                            ion_type_by.append(ion_type1)
                else:
                    ficharge=int(ion_type1.split("_")[2])
                    if  ficharge<=MAXFICHARGE:
                        ion_type_BY.append(ion_type1)
            # print(iden_pep)
            # print("ion_type_by",ion_type_by)
            # print("ion_type_BY",ion_type_BY)
            ion_type_len_by=len(ion_type_by)
            if ion_type_len_by==0:
                pass
            else:
                intensity=ion_1[1]/ion_type_len_by
                # print("intensity",intensity)
                for k in ion_type_by:
                    FragmentType=k[0]
                    FragmentNumber=int(k.split("_")[0][1:])
                    Fragmentcharge=int(k.split("_")[1])
                    if "loss" in k:
                        Fragmentloss=k.split("_")[-2].upper()
                        Fragmentloss=Fragmentloss.replace("H20","H2O")
                        Fragmentloss=lossdict_by[Fragmentloss]
                    else:
                        Fragmentloss=""
                    # print("Fragmentloss",Fragmentloss)
                    Fragmenthexnac=k.split("_")[-1]
                    Fragmenthexnac=lossdict_by[Fragmenthexnac]
                    # print("Fragmenthexnac",Fragmenthexnac)
                    ion_k=FragmentType+str(Fragmentcharge)+Fragmentloss+Fragmenthexnac
                    # print("ion_k",ion_k)
                    # print("ionname",ionname)
                    j = ionname_by.index(ion_k)
                    # print("j",j)
                    i = FragmentNumber - 1 if FragmentType == 'b' else length - FragmentNumber - 1
                    # print(i,j)
                    ions_by[i, j] = intensity 

            ion_type_len_BY=len(ion_type_BY)
            if ion_type_len_BY==0:
                pass
            else:
                intensity=ion_1[1]/ion_type_len_BY
                # print("intensity",intensity)
                for k in ion_type_BY:
                    FragmentType=k[0]
                    FragmentNumber=int(k.split("_")[1])
                    Fragmentcharge=int(k.split("_")[2])
                    if "loss" in k:
                        Fragmentloss=k.split("_")[-1].upper()
                        Fragmentloss=Fragmentloss.replace("H20","H2O")
                        Fragmentloss=lossdict_BY[Fragmentloss]
                    else:
                        Fragmentloss=""
                    # print("Fragmentloss",Fragmentloss)
                    # print("Fragmenthexnac",Fragmenthexnac)
                    ion_k=FragmentType+str(Fragmentcharge)+Fragmentloss
                    # print("ion_k",ion_k)
                    # print("ionname",ionname)
                    j = ionname_BY.index(ion_k)
                    # print("j",j)
                    i = FragmentNumber
                    # print(i,j)
                    # print(k)
                    ions_BY[i, j] = intensity
        # if not ions_by.any() or not ions_BY.any():
        #     print(GlySpec)
        #     # import ipdb
        #     # ipdb.set_trace()
        # else:
        outputdataframe.loc[len(outputdataframe)] = [GlySpec,sequence, decoration_intial,decoration_ACE,PlausibleStruct,charge, ions_by,ions_BY,iden_pep]
    outputdataframe.to_json(outputfile)
    print("finish")
    print(outputdataframe)

# ----------------------- irt prediction--------------------------------#
def report2irtmatrix(filename,outputfile):#####因为preprocess需要ions故这里产生的数据ions都用1代替，是无意义的
    # if isinstance(filename,str):
    #     file = filename
    #     path0 = os.getcwd()
    #     filepath = os.path.join(path0, file)
    #     testdata = pd.read_csv(filepath)
    # elif isinstance(filename,pd.DataFrame):
    #     testdata=filename
    #     if name:
    #         file=name
    #     else:
    #         file="rawdata.csv"
    # else:
    #     raise Exception
    testdata = pd.read_csv(filename)
    down = 0
    decorationdict={}
    decorationdict['PHO']=1
    decorationdict['GLY']=5
    #糖没有代替原来phos的位置，而是随机初始化了，希望可以同时处理糖和磷酸化，做后修饰 crosstalk
    #糖基化后续可以随机初始化或者复制磷酸化的赋值
    decorationdict['OXI']=2
    decorationdict['CAR'] = 3
    decorationdict['ACE'] = 4
    print(f"Columns for the input file {testdata.columns}. Please check it caters to the columns name in the code")
    # outputdata=testdata.drop_duplicates(subset=["PP.iRTEmpirical"]) #为什么要去重？
    zz = testdata[["GlySpec","iden_pep","Peptide",'Charge','RT']] #注意columns名字
    def min_max_scale(x, min_val, max_val):
        new_x = (x - min_val) / (max_val - min_val)
        return new_x
    min_val = zz["RT"].min()
    max_val = zz["RT"].max()
    zz["irt0"] = zz["RT"].apply(lambda x: min_max_scale(x, min_val, max_val))
    # ipdb.set_trace()
    zz=zz[["GlySpec","iden_pep","Peptide",'Charge','irt0']]
    zz.columns = ["GlySpec","iden_pep", "sequence", 'charge', 'irt']
    zz["sequence"]=zz["sequence"].str.replace("J","N")
    def decoration(row):
        iden_pep_list = masses.peptide_process(row["iden_pep"])
        decoration = iden_pep_list["modifications"]
        length = len(row["sequence"])
        decoration_intial = [0] * length
        decoration_ACE = [0] * length  # 把4单独存一个list，这样可以处理首位有两个修饰的情况
        for k in decoration:
            if k["type"] == "mod":
                name = k["name"].lstrip("(")
                name = name.rstrip(")1").upper()
                if name == "ACE":
                    decoration_ACE[0] = decorationdict[name]
                else:
                    site = k["position"]
                    decoration_intial[site] = decorationdict[name]
            if k["type"] == "glyco":
                site = k["position"]
                PlausibleStruct = k["structure"]
                decoration_intial[site] = decorationdict["GLY"]
                # ipdb.set_trace()
        return decoration_intial,decoration_ACE,PlausibleStruct
    result = zz.apply(decoration, axis=1).apply(pd.Series)
    zz['decoration'] = result[0]
    zz['decoration_ACE'] = result[1]
    zz['PlausibleStruct'] = result[2]
    zz['ions']=[1]*len(zz)
    fields = "GlySpec,sequence,decoration,decoration_ACE,PlausibleStruct,charge,ions,irt,iden_pep".split(',')
    zz=zz[fields]
    zz.to_json(outputfile)
    print("finish")

# ----------------------- command line------------------------------#
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DDAfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=True,
        help="DDA raw file",
    )
    parser.add_argument(
        "--outputfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=False,
        help="output filename",
    )
    parser.add_argument(
        "--do_ms2",
        action="store_true", help="do ms2",

    )
    parser.add_argument(
        "--do_irt",
        action="store_true", help="do irt",

    )
    parser.add_argument(
        "--do_BY",
        action="store_true", help="do BY",

    )
    parser.add_argument(
        "--do_byBY",
        action="store_true", help="do BY",

    )
    args = parser.parse_args()
    if args.do_ms2:
        if not args.outputfile:
            resultreport2matrix(args.DDAfile,args.DDAfile[:-4]+"_processed.json")
        else:
            resultreport2matrix(args.DDAfile,args.outputfile)
    if args.do_irt:
        report2irtmatrix(args.DDAfile,args.outputfile)
    if args.do_BY:
        if not args.outputfile:
            report2BYmatrix(args.DDAfile,args.DDAfile[:-4]+"_BY.json")
        else:
            report2BYmatrix(args.DDAfile,args.outputfile)
    if args.do_byBY:
        if not args.outputfile:
            byBYmatrix(args.DDAfile,args.DDAfile[:-4]+"_byBY.json")
        else:
            byBYmatrix(args.DDAfile,args.outputfile)
if __name__=='__main__':
    main()
    # report2irtmatrix(target)
