#只做谱图预测，对于已经训练好的模型，输入数据进行预测
from fastNLP import Tester
from preprocess import *
from transformers import BertConfig,RobertaConfig
from model_gly import *
from Bertmodel import ModelbyBYms2_bert
from utils import *
import os
from pathlib import Path
####################
set_seed(seed)
import argparse
import json
import numpy as np
# ----------------------- args ------------------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafold', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/Five_tissues/",
                        help='datafold ')    
    parser.add_argument('--trainpathcsv', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005413/PXD005413_MouseHeart_data_1st.csv",
                        help='the train csv for test')
    parser.add_argument('--bestmodelpath', type=str, 
                        default="/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/Five_tissues/Mouse_test_PXD005413_data_1st_combine_byBYprocessed/checkpoints/2023-04-13-00-45-52-259709/epoch-75_step-19800_mediancos-0.912608.pt",
                        help='the best model path')
    parser.add_argument('--device', type=int, default=1, help='cudadevice')
    parser.add_argument('--postprocessing', type=str, default="off", help='on/off')
    parser.add_argument('--savename', type=str, 
                        default="test_byBY_PXD005413_model_0.912608", help='the output file name')
    parser.add_argument('--ms2_method', type=str, 
                        default="cos_sqrt", help='metric')
    args = parser.parse_args()
    return args
args=parsering()
savefold= os.path.join(args.datafold, "test_replace_predict/")
if os.path.exists(savefold):
    pass
else:
    os.mkdir(savefold)
savename=args.datafold+"test_replace_predict/"+args.savename
device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
trainpathcsv=args.trainpathcsv
traindatajson=trainpathcsv[:-4]+"_byBYprocessed.json"
# ipdb.set_trace()
# traindatajson_path = Path(traindatajson)
# if traindatajson_path.exists():
#     print(f"{traindatajson} exists.")
#     # df_fp=pd.read_json(traindatajson)
# else:
print(f"{traindatajson} does not exist. Begin matrixwithdict to produce result...")
os.system("python matrixwithdict.py \
--do_byBY \
--DDAfile {} \
--outputfile {}".format(trainpathcsv,traindatajson))


# -----------------------------------------------------------#
fpath=traindatajson   #输入数据什么格式
databundle = PPeptidePipebyBY(vocab=vocab).process_from_file(paths=fpath)
totaldata=databundle.get_dataset("train")
print("totaldata",totaldata)


def encode_dataset(obj):
    if isinstance(obj, np.ndarray) or isinstance(obj,torch.Tensor):
        return obj.tolist()  # 将 Numpy 数组转换为列表
    else:
        print(type(obj))
        ipdb.set_trace()
        return str(obj)  # 对于无法序列化的对象，将其转换为字符串

def save_dataset_as_json(dataset, file_path):
    # 将数据集转换为字典格式
    data_dict = {}
    dataset_field=['GlySpec', "peptide","charge", 'ions_by', 'ions_BY', 'iden_pep', "ions_BY_p",'_id']
    for field_name in dataset_field:
        data_dict[field_name] = list(dataset.get_field(field_name))
    # 使用自定义编码器来处理非序列化对象
    encoded_dict = json.loads(json.dumps(data_dict, default=encode_dataset))
    # 将字典保存为 json 文件
    with open(file_path, 'w') as f:
        json.dump(encoded_dict, f, indent=4)

save_dataset_as_json(totaldata, args.datafold+"test_replace_predict/"+args.savename+"_byBY_testdata.json")
# import ipdb
# ipdb.set_trace()
###########model

config=BertConfig.from_pretrained("bert-base-uncased")
bestmodelpath=args.bestmodelpath
deepms2=ModelbyBYms2_bert(config)
bestmodel=torch.load(bestmodelpath).state_dict()
deepms2.load_state_dict(bestmodel,strict=False)
model_sign=bestmodelpath.split("/")[-1]
# ipdb.set_trace()
#model info
from torchinfo import summary
summary(deepms2)

from fastNLP import Const
metrics=Metric_byBY_outputmsms(savename=savename,pred=Const.OUTPUT,target=Const.TARGET,
                               pred_by="pred_by",pred_BY="pred_BY",target_by="target_by",target_BY="target_BY",seq_len='seq_len',
                               num_col=num_col,sequence='sequence',charge="charge",decoration="decoration",
                               peptide="peptide",PlausibleStruct="PlausibleStruct",
                               args=args)
from MSELoss_for_byBY import MSELoss_byBY
loss=MSELoss_byBY(pred_by="pred_by",pred_BY="pred_BY",target_by="target_by",target_BY="target_BY")

############tester


pptester=Tester(model=deepms2,device=device,data=totaldata,
                loss=loss,metrics=metrics,
                batch_size=BATCH_SIZE)
from timeit import default_timer as timer
train_time_start = timer()

pptester.test()


# import ipdb
# ipdb.set_trace()
train_time_end = timer()
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
total_train_time_model_2 = print_train_time(start=train_time_start,
                                        end=train_time_end,
                                        device=device)

# outputfile=processeddata[0:-4]+"."+model_sign+".model.csv"
# mgfinput=outputfile[0:-4]+"monomz.csv"
#这里可以load postprocessing_spectra.py把结果变成matrix

postprocessing=args.postprocessing
if postprocessing=="on":
    import postprocessing_spectra
    print("savename for postprocessing",savename)
    postprocessing_spectra.postprocessing(savename)
print("end")
