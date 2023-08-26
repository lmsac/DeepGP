import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import DataSet
import torch
import numpy as np
from fastNLP import BucketSampler
import pandas as pd
import json
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP import DataSet
from fastNLP import  Instance
from fastNLP import BucketSampler
from fastNLP import DataSetIter
from fastNLP.io import Loader
from fastNLP.io import Pipe
from fastNLP import Vocabulary
from fastNLP.io import DataBundle
def peptideload(fpath):##优化
    fields=["peptide","charge","ions","qvalue"]
    datadict={}
    for field in fields:
        datadict[field]=[]
    with open(fpath,'r') as f:
        data=json.load(f)
    for i in data:
        for field in fields:
            datadict[field].append(i[field])
    return DataSet(datadict)
class NPPeptideLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions", "qvalue"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with open(fpath, 'r') as f:
            data = json.load(f)
        for i in data:
            for field in fields:
                datadict[field].append(i[field])
        return DataSet(datadict)

    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass

class NPPeptidePipe(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions", "qvalue"]
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')


        return data_bundle



    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)
        ionsby=sorted(data_bundle.get_dataset('train')[0]["ions"].keys())
        tensorfields = ['peptide_tensor','ions_tensor']

        if self.vocab==None:
            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')
        #处理ions，用顺序排列转成矩阵
        def process_ions(ions):
            ions_tensor=[]
            for by in ionsby:
                ions_tensor.append(ions[by])
            ions_tensor=torch.Tensor(ions_tensor)
            max=ions_tensor.max().item()
            ions_tensor=ions_tensor/max

            return torch.log2(ions_tensor.transpose(0,1).reshape(-1,)+1)
        data_bundle.apply_field(process_ions,'ions','target')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input('peptide_tokens','peptide_length')
        data_bundle.set_target('target')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = NPPeptideLoader().load(paths)
        return self.process(data_bundle)
# ----------------------- pytorch dataset processing ------------------------------#
#从DeepFLR到Deepglyco目前只改了pytorch dataset processing
class PPeptideLoader(Loader):
    def __init__(self):
        super().__init__()


    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions","decoration","decoration_ACE","PlausibleStruct",'irt',"_id","GlySpec","iden_pep"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        # import ipdb
        # ipdb.set_trace()
        data=pd.read_json(fpath)
        datadict["peptide"]+=list(data["sequence"])
        datadict["charge"] += list(data["charge"])
        datadict["ions"] += list(data["ions"])
        datadict["decoration"] += list(data["decoration"])
        datadict["decoration_ACE"]+= list(data["decoration_ACE"])
        datadict["PlausibleStruct"]+= list(data["PlausibleStruct"])
        datadict["_id"]=range(len(datadict["peptide"]))
        datadict["GlySpec"]+= list(data["GlySpec"])
        datadict["iden_pep"]+= list(data["iden_pep"])
        
        try:
            datadict["irt"] += list(data["irt"])
            # print("DataSet(datadict)",DataSet(datadict))
            return DataSet(datadict)
        except:
            datadict["irt"]=range(len(datadict["ions"]))
            # print("data[sequence]",data["sequence"])
            # print("data[decoration]",data["decoration"])
            # print("data[PlausibleStruct]",data["PlausibleStruct"])
            # print("datadict[ions])",datadict["ions"])
            # print("datadict[irt]",datadict["irt"])
            # print("DataSet(datadict)2",DataSet(datadict))
            #有好几行都是一样的，因为他们糖结构不一样而这里没有包括，考虑糖结构是否会对by造成影响，然后看是否需要引入
            return DataSet(datadict)
    

    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass

class PPeptidePipe(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions","decoration","decoration_ACE","PlausibleStruct",'irt',"_id","GlySpec","iden_pep"]
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():
            # print("name",name)
            # print("dataset",dataset)
            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            # print('peptide_tokens',list(dataset['peptide_tokens']))
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')
            # print('peptide_length',list(dataset['peptide_length']))
        return data_bundle

    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)

        if self.vocab==None: 
            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')
        #处理ions，用顺序排列转成矩阵
        def process_ions(ions_tensor):
            ions_tensor=torch.Tensor(ions_tensor)
            max=ions_tensor.max().item()
            ions_tensor=ions_tensor/max
            #我在写DP计算的时候，有做归一化，这里还要做归一化吗，还有log2
            return torch.log2(ions_tensor.reshape(-1,)+1) #做了log2!!
        data_bundle.apply_field(process_ions,'ions','target')
        from masses import glyco_process
        def strct_token(instance):
            PlausibleStruct=instance["PlausibleStruct"]
            _,_,Struct_tokens,nodef=glyco_process(PlausibleStruct)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            import dgl
            graph=dgl.graph(Struct_tokens)
            graph.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            # print("Struct_tokens",Struct_tokens)
            return {"strct_graph":[graph]}
        data_bundle.apply_more(strct_token)
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            decoration_ACE=instance["decoration_ACE"]
            # print("peptide",instance["peptide"])
            # print("decoration",decoration)
            # print("peptide_tokens",peptide_tokens)
            for i,token in enumerate(peptide_tokens):
                # print("i",i)
                # print("token",token)
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    if decoration_ACE[i]!=0:
                        decoration_ids.append(decoration_ACE[i])
                    else:
                        decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            # print("input_ids",input_ids)
            # print("decoration_ids",decoration_ids)
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)

        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input("decoration_ids",'peptide_tokens','peptide_length','decoration',"decoration_ACE","PlausibleStruct",'charge',"input_ids","_id")
        data_bundle.set_target('target','irt')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')
        data_bundle.set_pad_val("decoration",0)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoader().load(paths)
        return self.process(data_bundle)

from fastNLP.core import Padder
class PadderforBY(Padder):
    def __init__(self,pad_val):
        r"""
        :param pad_val: int, pad的位置使用该index
        :param pad_length: int, 如果为0则取一个batch中最大的单词长度作为padding长度。如果为大于0的数，则将所有单词的长度
            都pad或截取到该长度.
        """
        super().__init__(pad_val=pad_val) 
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        padded_array=torch.cat(contents,dim=0)
        return padded_array
    
class PPeptideLoaderbyBY(Loader):
    def __init__(self):
        super().__init__()


    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions_by","ions_BY","decoration","decoration_ACE","PlausibleStruct",'irt',"_id","GlySpec","iden_pep"]
        with open(fpath) as data:
            datadict0 = json.load(data)
            new_keys = {'sequence': 'peptide'}
            datadict = {}
            for key in datadict0:
                if key in new_keys:
                    new_key = new_keys[key]
                else:
                    new_key = key
                if new_key in fields:
                    # import ipdb
                    # ipdb.set_trace()
                    datadict[new_key] = list(datadict0[key].values())
            datadict["_id"]=list(range(len(datadict["peptide"])))
            # import ipdb
            # ipdb.set_trace()
            try:
                datadict["irt"] += list(datadict0["irt"])
            except:
                # import ipdb
                # ipdb.set_trace()
                datadict["irt"]=list(range(len(datadict["peptide"])))
                print("no rt information contained in the data!")

        # datadict=pd.DataFrame(datadict)
        # datadict=datadict[datadict["charge"]==2]
        # print(len(datadict["peptide"]))
        # datadict = datadict.to_dict(orient='list')
        # import ipdb
        # ipdb.set_trace()

        return DataSet(datadict)

class PPeptidePipebyBY(Pipe):
    def __init__(self,vocab=None,args=None):
        self.fields=["peptide", "charge", "ions_by","ions_BY","decoration","decoration_ACE","PlausibleStruct",'irt',"_id","GlySpec","iden_pep"]
        self.vocab=vocab
        self.args=args
    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():
            # print("name",name)
            # print("dataset",dataset)
            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            # print('peptide_tokens',list(dataset['peptide_tokens']))
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')
            # print('peptide_length',list(dataset['peptide_length']))
        return data_bundle

    def process(self, data_bundle: DataBundle) -> DataBundle:
        if False:
            pass
        # if self.args.model_ablation not in ["BERT","YYglyco","DeepFLR"]:##using protbert
        #     from transformers import BertTokenizer
        #     tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert")
            
        else:
            self._tokenize(data_bundle)
        # import ipdb
        # ipdb.set_trace()
            if self.vocab==None: 
                vocab = Vocabulary(unknown='<unk>', padding='<pad>')
                import ipdb
                ipdb.set_trace()
            else:
                vocab=self.vocab
            vocab.from_dataset(data_bundle.get_dataset('train'),
                            field_name="peptide_tokens",
                            )
            vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
            data_bundle.set_vocab(vocab, 'peptide_tokens')
        #联立主要目的就是为了一起归一化，得到相对信息，因此max就是一行ions_by和ion_BY的最大值
        def process_ions(instance):
            ions_by=torch.Tensor(instance["ions_by"])
            ions_BY=torch.Tensor(instance["ions_BY"])
            maxi=max(ions_by.max().item(),ions_BY.max().item())
            ions_by=ions_by/maxi
            ions_BY=ions_BY/maxi
            ions_by=ions_by.reshape(-1,)
            # ions_BY=ions_BY.reshape(-1,)
            # ions_by=torch.log2(ions_by.reshape(-1,)+1)
            # ions_BY=torch.log2(ions_BY.reshape(-1,)+1)
            #不做log2了，后面计算相似度的时候一起处理
            return {"ions_by_p":ions_by,"ions_BY_p":ions_BY}
        data_bundle.apply_more(process_ions)
        from masses import glyco_process
        def strct_token(instance):
            PlausibleStruct=instance["PlausibleStruct"]
            _,_,Struct_tokens,nodef=glyco_process(PlausibleStruct)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            import dgl
            graph=dgl.graph(Struct_tokens)
            graph.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            # print("Struct_tokens",Struct_tokens)
            return {"strct_graph":[graph]}
        data_bundle.apply_more(strct_token)
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            decoration_ACE=instance["decoration_ACE"]
            # print("peptide",instance["peptide"])
            # print("decoration",decoration)
            # print("peptide_tokens",peptide_tokens)
            for i,token in enumerate(peptide_tokens):
                # print("i",i)
                # print("token",token)
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    if decoration_ACE[i]!=0:
                        decoration_ids.append(decoration_ACE[i])
                    else:
                        decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            # print("input_ids",input_ids)
            # print("decoration_ids",decoration_ids)
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)
        data_bundle.set_pad_val("decoration",0)
        # data_bundle.apply_field(lambda x: x.reshape(-1,16),"ions_BY_p","ions_BY_p")
        def count_edge(instance):
            return {"graph_edges":instance["ions_BY_p"].shape[0]}
        data_bundle.apply_more(count_edge)
        # import ipdb
        # ipdb.set_trace()
        data_bundle.set_input("decoration_ids",'peptide_tokens',"graph_edges",'peptide_length','decoration',
                              "decoration_ACE","PlausibleStruct",'charge',"input_ids","_id","peptide",
                                'ions_by_p',"ions_BY_p")
        data_bundle.set_target('irt',)
        data_bundle.set_pad_val("decoration",0)
        data_bundle.get_dataset('train').set_padder("ions_BY_p",PadderforBY(0))
        #这里set_padder还有用吗，毕竟target已经改了
        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoaderbyBY().load(paths)
        return self.process(data_bundle)



# ----------------------- not in use ------------------------------#
class PPeptidePipe_targetBY(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions","decoration","decoration_ACE","PlausibleStruct",'irt',"_id","GlySpec","iden_pep"]
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():
            # print("name",name)
            # print("dataset",dataset)
            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            # print('peptide_tokens',list(dataset['peptide_tokens']))
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')
            # print('peptide_length',list(dataset['peptide_length']))
        return data_bundle

    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)

        if self.vocab==None: 
            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')
        #处理ions，用顺序排列转成矩阵
        def process_ions(ions_tensor):
            ions_tensor=torch.Tensor(ions_tensor)
            max=ions_tensor.max().item()
            ions_tensor=ions_tensor/max
            #我在写DP计算的时候，有做归一化，这里还要做归一化吗，还有log2
            return torch.log2(ions_tensor.reshape(-1,)+1)
        data_bundle.apply_field(process_ions,'ions','target')
        # import sys
        # sys.path.append('..')
        from masses import glyco_process
        def strct_token(instance):
            PlausibleStruct=instance["PlausibleStruct"]
            _,_,Struct_tokens,nodef=glyco_process(PlausibleStruct)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            import dgl
            graph=dgl.graph(Struct_tokens)
            graph.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            # print("Struct_tokens",Struct_tokens)
            return {"strct_graph":[graph]}
        data_bundle.apply_more(strct_token)
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            decoration_ACE=instance["decoration_ACE"]
            # print("peptide",instance["peptide"])
            # print("decoration",decoration)
            # print("peptide_tokens",peptide_tokens)
            for i,token in enumerate(peptide_tokens):
                # print("i",i)
                # print("token",token)
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    if decoration_ACE[i]!=0:
                        decoration_ids.append(decoration_ACE[i])
                    else:
                        decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            # print("input_ids",input_ids)
            # print("decoration_ids",decoration_ids)
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)
        data_bundle.apply_field(lambda x: x.reshape(-1,16),"target","target")
        def count_edge(instance):
            
            return {"graph_edges":instance["target"].shape[0]}
        data_bundle.apply_more(count_edge)
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input("decoration_ids",'peptide_tokens','peptide_length','decoration',"decoration_ACE","PlausibleStruct",'charge',"input_ids","_id","peptide")
        data_bundle.set_target('target','irt',"graph_edges")
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')
        data_bundle.set_pad_val("decoration",0)
        data_bundle.get_dataset('train').set_padder("target",PadderforBY(0))
        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoader().load(paths)
        return self.process(data_bundle)


# from utils import *
# filename="20221113_pglyco_traincsv_testprocessed.json"
# databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=filename)
# totaldata=databundle.get_dataset("train")
# print("totaldata",totaldata)
# vocab=databundle.get_vocab("peptide_tokens")
# print("vocab",vocab)
############################################只有肽段和charge没有target的产生数据
class PPeptideLoader_notarget(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions","decoration",'irt']
        datadict = {}
        for field in fields:
            datadict[field] = []
        data=pd.read_csv(fpath)
        datadict["peptide"]+=list(data["key"])
        datadict["charge"] += list(data["PP.Charge"])
        ############下面三个是没有的，通过后续操作产生
        datadict["ions"] += range(len(data["key"]))
        datadict["decoration"] += list(data["key"])
        datadict["irt"]=range(len(data["key"]))
        return DataSet(datadict)


    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass
def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(presequence)
    sig=0
    if "4" in presequence:
        if presequence.index("4") !=0:
            presequence=presequence.replace('4','')
            presequence='4'+presequence
            # print(presequence)
    for i in range(l):
        if presequence[i]=='4':
            assert i==0
            sig=1
            phos.append(4)
            continue
        elif presequence[i] in number:
            phos[-1] = int(presequence[i])
            continue

        phos.append(0)
        if sig:
            sig=0
            phos.pop()
    return phos

def dropnumber(string:str):
    for i in range(5):
        string=string.replace(str(i),"")
    return string
def createzerotarget(peptide_tokens,num_col=36):
    return np.zeros(((len(peptide_tokens)-1)*num_col))
class PPeptidePipe_notarget(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions", "decoration",'irt']
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')


        return data_bundle



    def process(self, data_bundle: DataBundle) -> DataBundle:
        data_bundle.apply_field(countdecoration,'peptide','decoration')#decoration搞定
        data_bundle.apply_field(dropnumber, 'peptide', 'peptide')#peptide搞定
        self._tokenize(data_bundle)
        if self.vocab==None:

            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')



        def countphos(lis):
            return lis.count(1)
        data_bundle.apply_field(countphos,field_name='decoration',new_field_name='pnumber')
        data_bundle.apply_field(createzerotarget, field_name='peptide_tokens', new_field_name='target')
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            for i,token in enumerate(peptide_tokens):
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input("decoration_ids",'peptide_tokens','peptide_length','decoration','charge','pnumber',"input_ids")
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output

        data_bundle.set_target('target','irt')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')
        data_bundle.set_pad_val("decoration",0)
        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoader_notarget().load(paths)
        return self.process(data_bundle)
