import pandas as pd
import os
import numpy as np
from pathlib import Path
import masses
import mgf_processing
#1st step: format pglco3 result into training dataset (.csv) for subsequent processing (such as matrixwithdict)
# --------------------------- argparse ---------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--datafold', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005411/",
                        help='datafold ')
    parser.add_argument('--dfname', type=str, 
                        default="pGlycoDB-GP-FDR-Pro_PXD005411.txt",
                        help='pglyco3 crude result ')
    parser.add_argument('--mgfdatafold', type=str, 
                        default="MSConvert_mgf_PXD005411/" , 
                        help='mgf data fold')
    parser.add_argument('--output_name', type=str, 
                        default="PXD005411_MouseBrain_data_1st.csv", help='outputfile name')
    parser.add_argument('--dup', type=str,default="Drop_duplicated", help='Duplicated/Drop_duplicated/Retained_all')
    parser.add_argument('--mgfsourceorign', type=str,default="MsConvert", help='Please ensure the tool for producing mgf (MsConvert or pGlyco3)')
    args = parser.parse_args()
    return args
args=parsering()
DFNAME=args.datafold+args.dfname
mgfdatafold=args.datafold+args.mgfdatafold
output_name=args.datafold+args.output_name
only_duplicated=args.dup
mgfsourceorign=args.mgfsourceorign
assert mgfsourceorign in ["pGlyco3","MsConvert"], "mgfsourceorign not in [pGlyco3,MsConvert]"
# --------------------------- hyper paramaters ---------------------#
FRAG_AVA=["ETD","HCD_1","HCD_by","HCD_BY_2"]
FRAG_INDEX=[1,2] #"HCD_1" for BY prediction. "HCD_by" for by prediction
FRAG_MODE=[x for x in FRAG_AVA if FRAG_AVA.index(x) in FRAG_INDEX]
print(f"FRAG_MODE: {FRAG_MODE}")

jsonfold= os.path.join(mgfdatafold, "json/")
jsonname="SStruespectrum.json"
filter_jsonname="SStruespectrum_filtered_"+args.dup+".json" 
TOLER=20

# --------------------------- pglyco3 result processing---------------------#
def dea(row):
    if "Deamidated" in str(row["Mod"]):
        return "dea"
def pglyco3_result(DFNAME):
    df=pd.read_csv(DFNAME,sep="\t")
    # ipdb.set_trace()
    df["dea"]=df.apply(dea,axis=1)
    # ipdb.set_trace()
    df=df.loc[df["dea"]!="dea"]
    df.reset_index(inplace=True,drop=True)
    df_column=list(df.columns)
    print(f"Columns of df {df_column}",end = "\n\n")
    print(f"df rank should all be 1.. Please check!!: {list(df['Rank'].drop_duplicates())}",end = "\n\n")
    assert list(df['Rank'].drop_duplicates())==[1]
    df=df[["GlySpec","RawName","Scan", 'Charge',"Peptide","Mod",
           "PlausibleStruct",'GlySite',"RT","PrecursorMZ",
        'GlycanFDR','TotalFDR','Proteins']]
    #去掉RT和porecursor mz，从mgf中提取
    print(f"Row number of df {len(df)}",end = "\n\n")
    df.drop_duplicates(inplace=True)
    print(f"Row number of df after drop_duplicates {len(df)}",end = "\n\n")

    return df

def combine_iden_pep(instance):
    a=instance["Peptide"]
    b=instance["Mod"]
    e=""
    if not pd.isna(b):
        b=b.rstrip(";")
        for i in b.split(";"):
            for k in i.split(","):
                k=k[:3]+"."
                e+=k
        b=e
    else:
        b=None
    c=instance["GlySite"]-1  #GlySite 是从1开始的，会比index J 大一
    d=instance["Charge"]
    e=instance["PlausibleStruct"]
    return str(a)+"_"+str(b)+"_"+str(c)+"_"+str(d)+"_"+str(e)

def pglyco3_processing(df,
                    potential_glycosite_num=1,
                    only_duplicated="Drop_duplicated"):
    """Create required columns.

    Args:
    duplicated: True or False: whether or not only peak duplicated columns.
    True: only duplicated row are retained for repeatability test.
    False: only rows with lowest totalFDR for duplicated columns or unique columns are retained for training.
    
    """

    df["iden_pep"]=df.apply(combine_iden_pep,axis=1) #eg. JASQNQDNVYQGGGVCLDCQHHTTGINCER_16.Car.19.Car.28.Car._0_4_(N(N(H(H(H))(H(H)))))
    if only_duplicated=="Duplicated":
        df1=df[["iden_pep"]].loc[df["iden_pep"].duplicated()].drop_duplicates()
        df=df.loc[df["iden_pep"].isin(df1["iden_pep"])]
    print("Row number with multiple glycan sites: ",len(df.loc[df["Peptide"].str.count("J")>1]))
    # ipdb.set_trace()
    df=df.loc[df["Peptide"].str.count("J")==potential_glycosite_num] #必须含有的J为1 
    df.reset_index(inplace=True,drop=True)
    # ipdb.set_trace()
    if only_duplicated == "Drop_duplicated":
        df.sort_values(by='TotalFDR',ascending=True,inplace=True)
        # ipdb.set_trace()
        df.drop_duplicates(subset=['iden_pep'],inplace=True)
        df.reset_index(drop=True,inplace=True)
    if only_duplicated == "Retained_all":
        pass
    return df
# --------------------------- spectrum filtration---------------------#
#从json中找到相应的谱图，缩小搜索空间
def json_extraction(jsonfold=jsonfold,
                    jsonname=jsonname,
                    filename=filter_jsonname,
                    mgfsourceorign=mgfsourceorign):
    datalis=pd.read_json(os.path.join(jsonfold, jsonname))
    datalis["title"]=datalis["SourceFile"].map(str) + "-" + datalis["Spectrum"].map(str)
    datalis=datalis.loc[datalis["title"].isin(df["GlySpec"])]
    print("Please ensure the Spectrum numbers of MsConvert json files match those of the pGlyco3 result!")
    datalis.reset_index(inplace=True, drop=True)
    datalis.to_json(os.path.join(jsonfold, filename))
    return datalis
# ----------------------- ions picking ------------------------------#
def fragment_training(instance):
    spectrum=instance["GlySpec"]
    datalis_1=datalis.loc[datalis["title"]==spectrum]
    datalis_1=datalis_1.reset_index(drop=True)
    iden_pep=instance["iden_pep"]
    mz_calc=masses.pepfragmass(iden_pep,FRAG_MODE,3) #iden_pep已经改成了glysite，避免多J的可能
    ppm=TOLER
    FragmentMz=[]
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    # ipdb.set_trace()
    mass={"FragmentMz":FragmentMz}
    #FragmentMz：所有算出来的理论质荷比
    mzdict=mgf_processing.putTintensity(ppm, mass, datalis_1)
    for k in list(mzdict.keys()):
        if mzdict[k]==0:
            del mzdict[k]
    mzdict_1={}
    #补上mzdict的碎裂类型
    for i in mz_calc:
        for a in i:
            mz_calc_1=list(a.values())[0]
            if mz_calc_1 in list(mzdict.keys()):
                # print("a",a)
                # print("mzdict[mz_calc_1]",mzdict[mz_calc_1])
                type=list(a.keys())[0]
                intensity=mzdict[mz_calc_1]
                if not mz_calc_1 in mzdict_1.keys():
                    type_list=[]
                    type_list.append(type)
                    ions=(type_list,intensity)
                    mzdict_1[mz_calc_1]=ions
                else:
                    type_list=mzdict_1[mz_calc_1][0]
                    type_list.append(type)
                    ions=(type_list,intensity)
                    mzdict_1[mz_calc_1]=ions
    return mzdict_1

def mz_matching(instance):
    spectrum=instance["GlySpec"]
    datalis_1=datalis.loc[datalis["title"]==spectrum]
    datalis_1=datalis_1.reset_index(drop=True)
    iden_pep=instance["iden_pep"]
    mz_calc=masses.pepfragmass(iden_pep,["HCD_BY_2"],4)
    ppm=TOLER
    FragmentMz=[]
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    FragmentMz.sort()
    # ipdb.set_trace()
    mzexp=datalis_1["mz"][0]
    # mzexp=[round(num, 2) for num in mzexp]
    mzexp.sort()
    matchmz=[]
    for k in mzexp:
        i = (np.abs(np.array(FragmentMz) - k)).argmin()
        # ipdb.set_trace()
        if abs(FragmentMz[i] - k) < k * TOLER * 1 / 1000000: 
            matchmz.append(k)
    return {"matchmz":len(matchmz),"calc":len(FragmentMz),"mzexp":len(mzexp)}
# --------------------------- execution ---------------------#
if __name__=="__main__":
    DFNAME_path = Path(DFNAME)
    assert DFNAME_path.exists()
    # pglyco3 formatted result
    df_fp=pglyco3_result(DFNAME)
    df=pglyco3_processing(df_fp,
                        potential_glycosite_num=1,
                        only_duplicated=only_duplicated)
    df["Peptide"]=df["Peptide"].str.replace("J","N")
    df["GlySpec"]=df["RawName"].map(str) + "-" + df["Scan"].map(str)
    #json file
    json_path=Path(jsonfold,jsonname)
    if json_path.exists():
        print(f"{jsonname} exists.")
    else:
        print(f"{jsonname} does not exist. Begin mgf_process to produce required file...")
        datalis=mgf_processing.mgf_process(mgfdatafold=mgfdatafold,sourceorign=mgfsourceorign)
    file3_name_path = Path(jsonfold,filter_jsonname)
    datalis=json_extraction(jsonfold=jsonfold,
                jsonname=jsonname,
                filename=filter_jsonname,
                mgfsourceorign=mgfsourceorign)
    datalis["GlySpec"]=datalis["title"]
    df=pd.merge(df,datalis[["GlySpec",'RT_mgf', 'pepmass_mgf', 'charge_mgf']],on="GlySpec",how="left")
    df=df[[ "GlySpec",'Charge',"RT","PrecursorMZ","charge_mgf",'RT_mgf',"pepmass_mgf",'Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep',"TotalFDR"]] 
    df.drop_duplicates(subset=["GlySpec",'Charge',"RT","PrecursorMZ",'Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep',"TotalFDR"],inplace=True)
    df.reset_index(drop=True,inplace=True)
    assert df["RT_mgf"].isnull().any()==False 
    df["matching_ration"]=df.apply(mz_matching,axis=1)
    print("len(df_iden_pep.drop_duplicates())",len(df["iden_pep"].drop_duplicates()))
    print("len(df)",len(df))
    df["ions"]=df.apply(fragment_training,axis=1)
    df.to_csv(output_name,index=False)