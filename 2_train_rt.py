import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP.core.losses import LossBase
from preprocess import PPeptidePipe
from transformers import BertConfig,RobertaConfig
from model_gly import *
from Bertmodel import _2deepchargeModelms2_bert_irt
import pandas as pd
from pathlib import Path
from utils import *
# ----------------------- training time begin ------------------------------#
from timeit import default_timer as timer
train_time_start = timer()
import datetime
starttime = datetime.datetime.now()
print(f"starttime {starttime}",end="\n\n")
# ----------------------- model parameter for optimization ------------------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, 
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--warmupsteps', type=int, 
                        default=0,
                        help='warmupsteps')
    parser.add_argument('--weight_decay', type=float, 
                        default=1e-2,
                        help='weight_decay')
    parser.add_argument('--model_ablation', type=str, default="DeepFLR", help='model for ablation (BERT, DeepFLR, protein bert)')
    parser.add_argument('--device', type=int, default=4, help='cudadevice')
    parser.add_argument("--task_name",type=str, default="mouse_five_tissues")
    parser.add_argument('--testdata', type=str, default="alltest", help='data for test')
    parser.add_argument('--folder_path', type=str, 
                    default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/", 
                    help='the folder path for the training data')
    parser.add_argument("--trainpathcsv",type=str, 
                    default="Five_tissues/Mouse_five_tissues_data_rt_1st_combine.csv")
    parser.add_argument("--irt",type=str, default="no")
    parser.add_argument("--irt_csv",type=str, default="/remote-home/yxwang/test/DeepGP_code/data/mouse/All_adjust_irt.csv")
    parser.add_argument("--pattern",type=str,
                        default='*data_rt_1st.csv')
    parser.add_argument("--DeepFLR_modelpath",type=str,default="/remote-home/yxwang/test/DeepGP_code/model/DeepFLR/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399")
    args = parser.parse_args()
    return args
args=parsering()
lr=args.lr
warmupsteps=args.warmupsteps
weight_decay=args.weight_decay
testdata=args.testdata
pattern=args.pattern
irt_csv=args.irt_csv
DeepFLR_modelpath=args.DeepFLR_modelpath
device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
# ----------------------- model parameter------------------------------#
set_seed(seed)  
batch_size=128 #128
task_name=args.task_name
check_point_name=str(starttime)+"_rt_"+task_name+"_test_"+testdata+"_checkpoint"
print(f"Project name check_point_name {check_point_name} !",end="\n\n")
print(f"hyper parameter tested lr {lr} !",end="\n\n")
print(f"hyper parameter tested BATCH_SIZE {BATCH_SIZE} !",end="\n\n")
print(f"hyper parameter tested warmupsteps {warmupsteps} !",end="\n\n")
print(f"hyper parameter tested batch_size {batch_size} !",end="\n\n")
print(f"hyper parameter tested weight decay {weight_decay} !",end="\n\n")
# ----------------------- input pre-processing------------------------------#
folder_path = args.folder_path
import folder_walk
trainpathcsv_list=folder_walk.trainpathcsv_list(folder_path=folder_path,pattern=pattern)
print(f"Please check trainpathcsv_list {trainpathcsv_list}. The {len(trainpathcsv_list)} files contains all the training data!")
#将获得的文件合并以后并且导出，按照TotalFDR排序并去重
traincsv=pd.DataFrame()
trainpathcsv=folder_path+args.trainpathcsv
trainpathcsv_path = Path(trainpathcsv)
for x in  trainpathcsv_list:
    if  testdata in x:
        print(f"The test data is {x}! It is removed from the training data!")
    else:
        train=pd.read_csv(x)
        traincsv=pd.concat([traincsv,train])
        traincsv.sort_values(by='TotalFDR',ascending=True,inplace=True)
        traincsv.drop_duplicates(subset=['iden_pep'],inplace=True)
        traincsv.reset_index(drop=True,inplace=True)
        print(f"with the addition of {x}, the combined file contains {len(traincsv)} lines")
if args.irt=="yes":
    print("begin rt adjustment!")
    irt=pd.read_csv(irt_csv)
    print(traincsv.columns)
    traincsv.rename(columns={'RT': 'rt_old'}, inplace=True)
    traincsv["run"]=traincsv["GlySpec"].apply(lambda x: x.split("-")[0])
    traincsv=pd.merge(traincsv,irt,on=["run","iden_pep"],how="left")
    traincsv=traincsv[['GlySpec', 'Charge', 'rt_new' ,'Peptide', 'Mod', 'PlausibleStruct',\
         'GlySite', 'iden_pep', 'TotalFDR', 'PrecursorMZ' ]]
    traincsv.rename(columns={'rt_new': 'RT'}, inplace=True)
    traincsv.to_csv(trainpathcsv,index=False)
else:
    traincsv.to_csv(trainpathcsv,index=False)
traindatajson=trainpathcsv[:-4]+"processed_onlyirt.json"
traindatajson_path = Path(traindatajson)
print("Begin matrixwithdict to produce result...")
os.system("python matrixwithdict.py \
--do_irt \
--DDAfile {} \
--outputfile {}".format(trainpathcsv,traindatajson))


#traindata
filename=traindatajson
databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=filename)
totaldata=databundle.get_dataset("train")
print("totaldata",totaldata)
vocab=databundle.get_vocab("peptide_tokens")

traindata,devdata=totaldata.split(0.1)
def savingFastnlpdataset_DataFrame(dataset):
    dataset_field=dataset.field_arrays.keys()
    frame=pd.DataFrame(columns=dataset_field)
    for i in range(len(dataset)):
        c_list=[]
        for name in dataset_field:
            target=dataset.field_arrays[name][i]
            if name=="target":
                c_list.append(target.cpu().numpy().tolist())
            else:
                c_list.append(target)
        frame.loc[i]=c_list
    return frame
# ipdb.set_trace()
# devframe=savingFastnlpdataset_DataFrame(devdata)
# devframe.to_json("20230127_test_model_validata.json")
# torch.save(devframe,"20230127_test_model_validata")

# trainframe=savingFastnlpdataset_DataFrame(traindata)
# trainframe.to_json("20230127_test_model_train_data.json")
# torch.save(trainframe,"20230127_test_model_train_data")
# ipdb.set_trace()


# ----------------------- model ------------------------------#
model_ablation=args.model_ablation #DeepFLR
print(f"hyper parameter tested model_ablation {model_ablation} !",end="\n\n")
if model_ablation=="BERT":
    pretrainmodel="bert-base-uncased"  
    deepms2=_2deepchargeModelms2_bert_irt.from_pretrained(pretrainmodel)
    
if model_ablation=="DeepFLR":
    config=BertConfig.from_pretrained("bert-base-uncased")
    bestmodelpath=DeepFLR_modelpath
    model_sign=bestmodelpath.split("/")[-1]
    deepms2=_2deepchargeModelms2_bert_irt(config)
    bestmodel=torch.load(bestmodelpath).state_dict()
    origin_model=deepms2.state_dict()
    for key in origin_model.keys():
        if key in bestmodel.keys():
            if bestmodel[key].shape !=origin_model[key].shape:
                origin_model[key]=bestmodel[key][:origin_model[key].shape[0],] #linear尺寸匹配不上的部分进行裁剪
                print(f"size different key: {key}")
            else:
                origin_model[key]=bestmodel[key]
        else:
            print(f"not found key: {key}")  #GNN匹配不是的参数就保持原样
    deepms2.load_state_dict(origin_model)

#model info
import torchinfo
from torchinfo import summary
summary(deepms2)
# ----------------------- Trainer ------------------------------#
from fastNLP import Const
metrics=CossimilarityMetricfortest_outputrt(savename=None,pred="predirt",target="irt",seq_len='seq_len',
                                        num_col=num_col,sequence='sequence',charge="charge",
                                        decoration="decoration")
from fastNLP import MSELoss
loss=MSELoss(pred="predirt",target="irt")
import torch.optim as optim
optimizer=optim.AdamW(deepms2.parameters(),lr=lr,weight_decay=weight_decay)
from fastNLP import WarmupCallback,SaveModelCallback
save_path=filename[:-5]+"/checkpoints"
callback=[WarmupCallback(warmupsteps)]
callback.append(WandbCallback(project="Deepsweet",name=check_point_name,config={"lr":lr,"seed":seed,
"Batch_size":BATCH_SIZE,"warmupsteps":warmupsteps,"temperature":None,"weight_decay":None}))
callback.append(SaveModelCallback(save_path,top=3))
#trainer
from fastNLP import Trainer


if vocab_save:
    vocab.save(os.path.join(save_path,"vocab"))

pptrainer=Trainer(model=deepms2,    train_data=traindata,
                    device=device,  dev_data=devdata,
                save_path=save_path,
                  loss=loss,metrics=metrics,callbacks=callback,
                   optimizer=optimizer,n_epochs=N_epochs,batch_size=batch_size,update_every=int(BATCH_SIZE/batch_size),dev_batch_size=batch_size)
pptrainer.train()


# ----------------------- Time ------------------------------#
train_time_end = timer()
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
total_train_time_model_2 = print_train_time(start=train_time_start,
                                           end=train_time_end,
                                           device=device)