import pandas as pd
import numpy as np
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--inputfile1",
        default="DeepGP_result_predictmonomz.csv",
        type=str,
        required=False,
        help="inputfile: DeepGP output ",
    )
parser.add_argument(
        "--inputfile2",
        default="pGlycoDB-GP-FDR-Pro_fissionyeast.txt",
        type=str,
        required=False,
        help="inputfile: pGlyco3 output ",
    )
parser.add_argument(
        "--outputfile",
        default="output.csv",
        type=str,
        required=False,
        help="output file name ",
    )
parser.add_argument(
        "--Dataset",
        default="Yeast",
        type=str,
        required=False,
        help="Yeast/Mouse",
        choices=["Yeast","Mouse"]
    )

args = parser.parse_args()
inputfile1=args.inputfile1
inputfile2=args.inputfile2
outputfile=args.outputfile
Dataset=args.Dataset

def combine_iden_pep(instance):
    a=instance["Peptide"]
    b=instance["Mod"]
    e=""
    if not pd.isna(b) and len(b)!=0:
        b=b.rstrip(";")
        for i in b.split(";"):
            for k in i.split(","):
                k=k[:3]+"."
                e+=k
        b=e
    else:
        b=None
    c=instance["GlySite"]-1 
    d=instance["Charge"]
    e=instance["PlausibleStruct"]
    return str(a)+"_"+str(b)+"_"+str(c)+"_"+str(d)+"_"+str(e)

df=pd.read_csv(inputfile1)
df.rename(columns={"GlySpec":"SourceFile","_id":"id"},inplace=True)
df["PlausibleStruct"]=df["iden_pep"].apply(lambda x: x.split("_")[-1])
dfpglyco=pd.read_csv(inputfile2,sep="\t")
dfpglyco=dfpglyco.loc[dfpglyco["Peptide"].str.count("J")==1]
dfpglyco["SourceFile"]=dfpglyco["RawName"]+"-"+dfpglyco["Scan"].map(str)
dfpglyco["iden_pep"]=dfpglyco.apply(combine_iden_pep,axis=1)
dfpglyco=dfpglyco[["SourceFile","Mod","iden_pep","TotalScore","Proteins","GlycanComposition"]]
df=pd.merge(df,dfpglyco,on=["SourceFile","iden_pep"])
df["metriccombine"]=df["metric"]/max(df["metric"])+df["TotalScore"]/max(df["TotalScore"])

dftarget=df["metriccombine"].apply(lambda x:round(x,3))
dftarget=dftarget.drop_duplicates()
dftarget=dftarget.sort_values(ascending=True)
dftarget = np.array(dftarget) 
dftarget = dftarget.tolist()
b=1
out = pd.DataFrame(columns=["Combined_score","#Decoy/#PSMs","#PSMs"])
down=0

for a in dftarget:
    dfcutoff=df[df["metriccombine"]>=a]
    if len(dfcutoff)==0:
        break
    if Dataset=="Yeast":
        dfglycan_decoy=dfcutoff[dfcutoff["GlycanComposition"].str.contains("F")|dfcutoff["GlycanComposition"].str.contains("A")|dfcutoff["GlycanComposition"].str.contains("G")]
        dfglycan_Ndecoy=dfcutoff[~(dfcutoff["GlycanComposition"].str.contains("F")|dfcutoff["GlycanComposition"].str.contains("A")|dfcutoff["GlycanComposition"].str.contains("G"))]
        assert(len(dfcutoff)==len(dfglycan_decoy)+len(dfglycan_Ndecoy))
        dfglycan_Ndecoy=dfglycan_Ndecoy[~dfglycan_Ndecoy["GlycanComposition"].str.contains(re.escape("N(2)"))]
        FLR_glycan=(len(dfglycan_decoy)+len(dfglycan_Ndecoy))/(len(dfcutoff))
    elif Dataset=="Mouse":
        dftruecutoffmax=dfcutoff[dfcutoff["SourceFile"].str.contains("target")]
        FLR_glycan=(len(dfcutoff)-len(dftruecutoffmax))/(len(dfcutoff))
    else:
        print("Dataset should be within Yeast&Mouse.")
    if b<=FLR_glycan:
        FLR_glycan=b
    b = min(b, FLR_glycan)
    PSMs=int(len(dfcutoff))
    out.loc[down] = [a,FLR_glycan,PSMs]
    down+=1
    if  FLR_glycan==0:
        break
out.to_csv(outputfile,index=False)
