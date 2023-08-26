from fastNLP import Tester
from preprocess import *
from Bertmodel import _2deepchargeModelms2_bert_irt
from transformers import BertConfig,RobertaConfig
from model_gly import *
import os
from pathlib import Path
from utils import *
set_seed(seed)
# ----------------------- args ------------------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafold', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005413/",
                        help='datafold ')    
    parser.add_argument('--trainpathcsv', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005413/PXD005413_MouseHeart_data_rt_1st.csv",
                        help='the train csv for test')
    parser.add_argument('--bestmodelpath', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/Five_tissues/Mouse_five_tissues_data_rt_1st_combineprocessed_onlyirt/checkpoints/2023-03-21-13-06-24-960779/epoch-49_step-13818_mediancos-0.977327.pt",
                        help='the best model path')
    parser.add_argument('--device', type=int, default=0, help='cudadevice')
    parser.add_argument('--savename', type=str, 
                        default="test_rt_PXD005413_model_0.977327", help='the output file name')
    args = parser.parse_args()
    return args
args=parsering()
savename = os.path.join(args.datafold, "test_replace_predict/")
if os.path.exists(savename):
    pass
else:
    os.mkdir(savename)
savename=args.datafold+"test_replace_predict/"+args.savename
device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
# -----------------------------------------------------------#
trainpathcsv=args.trainpathcsv
traindatajson=trainpathcsv[:-4]+"processed_onlyirt.json"
traindatajson_path = Path(traindatajson)
print("Begin matrixwithdict to produce result...")
os.system("python matrixwithdict.py \
--do_irt \
--DDAfile {} \
--outputfile {}".format(trainpathcsv,traindatajson))
fpath=traindatajson
#现在matrixwithdict还没有存_id,要改一下
target="yes"
if target=="no_target":
    databundle=PPeptidePipe_notarget(vocab=vocab).process_from_file(paths=fpath)
    #PPeptidePipe_notarget还没有改成糖的样子
else:
    databundle = PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
totaldata=databundle.get_dataset("train")
print("totaldata",totaldata)
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
# testdata=savingFastnlpdataset_DataFrame(totaldata)
# testdata.to_json(args.datafold+"test_replace_predict/"+args.savename+"_testdata.json")
###########model

config=BertConfig.from_pretrained("bert-base-uncased")
bestmodelpath=args.bestmodelpath
deepms2=_2deepchargeModelms2_bert_irt(config)
bestmodel=torch.load(bestmodelpath).state_dict()
deepms2.load_state_dict(bestmodel,strict=False)
model_sign=bestmodelpath.split("/")[-1]

#model info
import torchinfo
from torchinfo import summary
summary(deepms2)

from fastNLP import Const
metrics=CossimilarityMetricfortest_predrt(savename=savename,pred="predirt",target="irt",seq_len='seq_len',
                                        num_col=num_col,sequence='sequence',charge="charge",
                                        decoration="decoration",_id="_id")
from fastNLP import MSELoss
loss=MSELoss(pred="predirt",target="irt")

############tester


pptester=Tester(model=deepms2,device=device,data=totaldata,
                loss=loss,metrics=metrics,
                batch_size=BATCH_SIZE)
from timeit import default_timer as timer
train_time_start = timer()

pptester.test()

train_time_end = timer()
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
total_train_time_model_2 = print_train_time(start=train_time_start,
                                        end=train_time_end,
                                        device=device)

print("end")