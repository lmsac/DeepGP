import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
# import os
import pandas as pd
import numpy as np
# import json
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP.core.losses import LossBase
from preprocess import NPPeptidePipe,PPeptidePipe
from torch.nn import CosineSimilarity
import ipdb
from utils import *
from sklearn.metrics import r2_score
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def attentionmask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask,不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
        #
        # >>> seq_len = torch.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len)
        # >>> print(mask.size())
        # torch.Size([14, 15])
        # >>> seq_len = np.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len)
        # >>> print(mask.shape)
        # (14, 15)
        # >>> seq_len = torch.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len, max_len=100)
        # >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask
class CossimilarityMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None, seq_len=None,num_col=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.num_col=num_col

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)
        L_1=pred.size(1)
        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        else:
            masks = None


        cos=CosineSimilarity(dim=1)
        if masks is not None:
            pred=pred.masked_fill(masks.eq(False), 0)
            self.cos += torch.sum(cos(pred,target)).item()
            self.total += pred.size(0)
            self.bestcos=max(self.bestcos,torch.max(cos(pred,target)).item())
            self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
        else:

            self.cos += torch.sum(cos(pred,target)).item()
            self.total += pred.size(0)
            self.bestcos = max(self.bestcos, torch.max(cos(pred,target)).item())
            self.listcos +=cos(pred, target).reshape(N,).cpu().numpy().tolist()
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """

        evaluate_result = {'mediancos':round(np.median(self.listcos),6),

                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class CossimilarityMetricfortest(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self,savename, pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration,_id="_id")
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nan=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,
                 charge=None,decoration=None,_id=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)
        L_1=pred.size(1)
        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        else:
            masks = None
        cos=CosineSimilarity(dim=1)
        #可以开根号，或者correlation coefficient
        
        # self.id_list+=_id.cpu().numpy().tolist()
        if masks is not None:
            s=torch.sum(cos(pred, target)).item()
            pred=pred.masked_fill(masks.eq(False), 0)
            if math.isnan(s):
                ipdb.set_trace()
                self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
                self.nan+=1

            else:

                self.cos += torch.sum(cos(pred,target)).item()
                self.total += pred.size(0)
                self.bestcos=max(self.bestcos,torch.max(cos(pred,target)).item())
                self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
            
        else:
            s=torch.sum(cos(pred, target)).item()
            print(s)
            if math.isnan(s):
                # ipdb.set_trace()
                print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
                self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
                self.nan+=1

            else:

                self.cos += s
                self.total += pred.size(0)
                self.bestcos = max(self.bestcos, torch.max(cos(pred,target)).item())
                self.listcos +=cos(pred, target).reshape(N,).cpu().numpy().tolist()
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        data=pd.Series(self.listcos)
        mediancos=np.median(self.listcos)
        # if mediancos>self.bestanswer:
        #     data.to_csv(self.savename+"Cossimilaritylist.csv",index=False)
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'mediancos':round(mediancos,6),
                            'nan number':self.nan,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class CossimilarityMetricfortest_outputmsms(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, savename,pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration,_id=_id)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nj=0
        self.savename=savename if savename else ""
        self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','ms2',"cos","id"])
        self.numcol=num_col
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0) #batch size
        L=pred.size(1) 
        # if seq_len is not None and target.dim() > 1:
        #     max_len = target.size(1)
        #     masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        # else:
        cos=CosineSimilarity(dim=1)
        s = torch.sum(cos(pred, target)).item()
        self.cos += s
        self.total += pred.size(0)
        self.bestcos = max(self.bestcos, torch.max(cos(pred, target)).item())
        self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
        # print(_id)
        # import ipdb
        # ipdb.set_trace()
        self.id_list+=_id.cpu().numpy().tolist()
        for i in range(N):
            il = seq_len[i]
            isequence=sequence[i][:il]
            icharge=charge[i]
            # ipdb.set_trace()
            idecoration=decoration[i][:il]
            ims2=pred[i].reshape((-1,self.numcol))[:il-1,:self.numcol]
            icos=self.listcos[self.nj]
            # ipdb.set_trace()
            self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         idecoration.cpu().numpy().tolist(),
                                         ims2.cpu().numpy().tolist(),
                                         icos,
                                         self.id_list[self.nj]]
            self.nj+=1


    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        # data=pd.Series(self.listcos)
        # data.to_csv("Cossimilaritylist.csv",index=False)
        # self.repsequence.to_json(self.savename+"result.json")
        self.repsequence.to_csv(self.savename+"_by_result.csv",index=False)
        evaluate_result = {'mediancos':round(np.median(self.listcos),6),
                            'total number':self.nj,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
            self.id_list=[]
        return evaluate_result

class CossimilarityMetricfortest_BY(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self,savename, pred=None, target=None, seq_len=None,
                 num_col=None,sequence=None,charge=None,decoration=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration,_id="_id",graph_edges="graph_edges")
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nan=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,
                 decoration=None,_id=None,graph_edges=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)  #改输入格式
        cos=CosineSimilarity(dim=0)  
        #现在的cos就是这样的，也没有开根号，因为target和pred都拍平了，这里不用dim=1
        #可以开根号，或者correlation coefficient
        # self.id_list+=_id.cpu().numpy().tolist()

        pred[pred<0]=0
        pred = torch.split(pred, graph_edges.tolist())
        target= torch.split(target, graph_edges.tolist())
        for c in range(len(graph_edges)):
            self.listcos+=[cos(pred[c].flatten(), target[c].flatten()).cpu().numpy()]
        s=sum(self.listcos)
        if math.isnan(s):
            print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
            self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
            self.nan+=1

        else:
            self.cos = s
            self.total += len(graph_edges)
            self.bestcos =  max(self.listcos).item()
            assert len(self.listcos)==self.total,"the length of cos list is different from the number of graphs"     
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        # data=pd.DataFrame(self.listcos)
        # data.columns=["cs"]
        mediancos=np.median(self.listcos)
        # if mediancos>self.bestanswer:
        #     data.to_csv(self.savename+"Cossimilaritylist.csv",index=False)
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'mediancos':round(mediancos,6),
                            'nan number':self.nan,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class CossimilarityMetricfortest_outputmsmsBY(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, savename,pred=None, target=None, seq_len=None,
                 num_col=None,sequence=None,charge=None,decoration=None,_id=None,
                 peptide=None,PlausibleStruct=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """
        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,
                             sequence=sequence,charge=charge,decoration=decoration,
                             _id="_id",graph_edges="graph_edges",
                             peptide=peptide,PlausibleStruct=PlausibleStruct)
        #增加了“graph_edges”
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nan=0
        self.nj=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
        self.repsequence=pd.DataFrame(columns=['repsequence','charge',
                                               "ipeptide","iPlausibleStruct",
                                               'ms2',"cos","id"])

    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,
                 decoration=None,_id=None,graph_edges=None,
                 peptide=None,PlausibleStruct=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)  #改输入格式
        cos=CosineSimilarity(dim=0)  
        pred[pred<0]=0
        pred = torch.split(pred, graph_edges.tolist())
        target= torch.split(target, graph_edges.tolist())
        for c in range(len(graph_edges)):
            self.listcos.append(cos(pred[c].flatten(), target[c].flatten()).cpu().numpy())
        s=sum(self.listcos)
        if math.isnan(s):
            print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
            self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
            self.nan+=1

        else:
            self.cos = s
            self.total += len(graph_edges)
            self.bestcos =  max(self.listcos).item()
            assert len(self.listcos)==self.total,"the length of cos list is different from the number of graphs"
            self.id_list+=_id.cpu().numpy().tolist()
        for i in range(len(graph_edges)):
            il = seq_len[i]
            isequence=sequence[i][:il]
            icharge=charge[i]
            ipeptide=peptide[i]
            iPlausibleStruct=PlausibleStruct[i]
            ims2=pred[i]
            icos=self.listcos[self.nj]
            # import ipdb
            # ipdb.set_trace()
            self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         ipeptide.tolist(),
                                         iPlausibleStruct.tolist(),
                                         ims2.cpu().numpy().tolist(),
                                         icos.tolist(),
                                         self.id_list[self.nj]]
            self.nj+=1


    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        data=pd.DataFrame(self.listcos)
        data.columns=["cs"]
        mediancos=np.median(self.listcos)
        if mediancos>self.bestanswer:
            pass
            # data.to_csv(self.savename+"Cossimilaritylist.csv",index=False)
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'mediancos':round(mediancos,6),
                            'nan number':self.nan,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        self.repsequence.to_csv(self.savename+"_BY_result.csv",index=False)
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

# --------------------------- byBY ---------------------#


def pearsonr(
        x,
        y,
        batch_first=True,
):
    r"""Computes Pearson Correlation Coefficient across rows.

    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:

    .. math::

        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}

    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.

    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`

    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`

    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:

        .. math::

            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2

        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.

    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])

    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr
def simlarcalc(spectrum_1_intensity,spectrum_2_intensity,type): 
    #提供两种方法，开根号的cosine similarity与correlation coefficient  "cos" or "corre"
    # 开根号也可以用poisson GL代替
    if type =="cos":
        cos = torch.nn.CosineSimilarity(dim=1)
        sim = cos(spectrum_1_intensity, spectrum_2_intensity)
    if type=="pcc":
        sim = pearsonr(spectrum_1_intensity, spectrum_2_intensity).squeeze()
    if type=="cos_sqrt":
        cos = torch.nn.CosineSimilarity(dim=1)
        sim = cos(torch.sqrt(spectrum_1_intensity), torch.sqrt(spectrum_2_intensity))
    if type=="corre_sqrt":
        spectrum_1_intensity=spectrum_1_intensity.sqrt()
        spectrum_2_intensity=spectrum_2_intensity.sqrt()
        sim = pearsonr(spectrum_1_intensity, spectrum_2_intensity).squeeze()
    return sim

class CossimilarityMetricfortest_byBY(MetricBase):

    def __init__(self,savename, pred=None, target=None, seq_len=None,
                 num_col=None,sequence=None,charge=None,decoration=None,
                 args=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,
                             charge=charge,decoration=decoration,_id="_id",graph_edges="graph_edges")
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nan=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
        self.args=args
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,
                 decoration=None,_id=None,graph_edges=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)  #改输入格式
        #simlarcalc:cos,corre,cos_sqrt,corre_sqrt
        # cos=CosineSimilarity(dim=0)  
        #现在的cos就是这样的，也没有开根号，因为target和pred都拍平了，这里不用dim=1
        #可以开根号，或者correlation coefficient
        # self.id_list+=_id.cpu().numpy().tolist()
        # pred = torch.split(pred, graph_edges.tolist())
        # target= torch.split(target, graph_edges.tolist())
        # L_1=pred.size(1)
        pred[pred<0]=0
        self.listcos+=simlarcalc(pred, target,self.args.ms2_method).cpu().numpy().tolist()
        # import ipdb
        # ipdb.set_trace()
        s=sum(self.listcos)
        #这里mask要改掉，因为超过序列部分的数据不能直接被mask掉，合并了
        
        self.id_list+=_id.cpu().numpy().tolist()
        if math.isnan(s):
            print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
            self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
            self.nan+=1

        else:
            self.cos = s
            self.total += N
            self.bestcos =  max(self.bestcos,max(self.listcos))
            assert len(self.listcos)==self.total,"the length of cos list is different from the number of graphs"     
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        # data=pd.DataFrame(self.listcos)
        # data.columns=["cs"]
        # import ipdb
        # ipdb.set_trace()
        mediancos=np.median(self.listcos)
        # if mediancos>self.bestanswer:
        #     data.to_csv(self.savename+"Cossimilaritylist.csv",index=False)
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'mediancos':round(mediancos,6),
                            'nan number':self.nan,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           "metric":self.args.ms2_method
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class Metric_byBY_outputmsms(MetricBase):

    def __init__(self,savename, pred=None, target=None, 
                 pred_by=None,pred_BY=None,
                 target_by=None,target_BY=None,seq_len=None,
                 num_col=None,sequence=None,charge=None,decoration=None,
                 _id=None,peptide=None,PlausibleStruct=None,
                 args=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, 
                             pred_by=pred_by,pred_BY=pred_BY,target_by=target_by,target_BY=target_BY,
                             seq_len=seq_len,sequence=sequence,
                             charge=charge,decoration=decoration,_id="_id",graph_edges="graph_edges",
                             peptide=peptide,PlausibleStruct=PlausibleStruct)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.listcosBY=[]
        self.nan=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
        self.nj=0
        self.repsequence=pd.DataFrame(columns=['repsequence','charge',
                                               "ipeptide","iPlausibleStruct",
                                               'ms2by',"ms2BY","metric","metricBY_cos","id"])
        # self.repsequence=pd.DataFrame(columns=['repsequence','charge',
        #                                        "ipeptide","iPlausibleStruct",
        #                                        'ms2by',"ms2BY","metric","id"])
        self.args=args
    def evaluate(self, pred, target, pred_by=None,pred_BY=None,
                 target_by=None,target_BY=None,
                 seq_len=None,sequence=None,charge=None,
                 decoration=None,_id=None,graph_edges=None,
                 peptide=None,PlausibleStruct=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)  #改输入格式
        #simlarcalc:cos,corre,cos_sqrt,corre_sqrt
        # cos=CosineSimilarity(dim=0)  
        #现在的cos就是这样的，也没有开根号，因为target和pred都拍平了，这里不用dim=1
        #可以开根号，或者correlation coefficient
        # self.id_list+=_id.cpu().numpy().tolist()
        # pred = torch.split(pred, graph_edges.tolist())
        # target= torch.split(target, graph_edges.tolist())
        # L_1=pred.size(1)
        pred[pred<0]=0
        pred_BY[pred_BY<0]=0
        self.listcos+=simlarcalc(pred, target,self.args.ms2_method).cpu().numpy().tolist()
        # ipdb.set_trace()

        pred_BY = torch.split(pred_BY, graph_edges.tolist()) 
        target_BY= torch.split(target_BY, graph_edges.tolist()) 
        for c in range(len(graph_edges)):
            # self.listcos+=[cos(pred[c].flatten(), target[c].flatten()).cpu().numpy()]
            cos=CosineSimilarity(dim=0)  
            
            self.listcosBY+=[cos(pred_BY[c].flatten(), target_BY[c].flatten()).cpu().numpy()]
            # print(self.listcosBY)
            # ipdb.set_trace()

        # ipdb.set_trace()
        # print("self.listcos",len(self.listcos))
        # import ipdb
        # ipdb.set_trace()
        if any(math.isnan(x) for x in self.listcos):
            print('The list contains nan values.')
            # ipdb.set_trace()
            self.listcos = [0 if math.isnan(x) else x for x in self.listcos]
            print(self.listcos.count(0))
        s=sum(self.listcos)
        #这里mask要改掉，因为超过序列部分的数据不能直接被mask掉，合并了
        
        self.id_list+=_id.cpu().numpy().tolist()
        if math.isnan(s):
            # ipdb.set_trace()
            print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
            self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
            self.nan+=1

        else:
            self.cos = s
            self.total += N
            self.bestcos =  max(self.bestcos,max(self.listcos))
            assert len(self.listcos)==self.total,"the length of cos list is different from the number of graphs"   
        masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        pred_by=pred_by.masked_fill(masks.eq(False), 0)
        for i in range(len(graph_edges)):
            il = seq_len[i]
            isequence=sequence[i][:il]
            icharge=charge[i]
            ipeptide=peptide[i]
            iPlausibleStruct=PlausibleStruct[i]
            # import ipdb
            # ipdb.set_trace()
            ims2by=pred_by[i].reshape((-1,(self.num_col)))[:il-1,:(self.num_col)]
            ims2BY=pred_BY[i]
            icos=self.listcos[self.nj]
            icosBY=self.listcosBY[self.nj]
            self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         ipeptide.tolist(),
                                         iPlausibleStruct.tolist(),
                                         ims2by.cpu().numpy().tolist(),
                                         ims2BY.cpu().numpy().tolist(),
                                         icos,
                                         icosBY,
                                         self.id_list[self.nj]]
            # self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
            #                              icharge.cpu().numpy().tolist(),
            #                              ipeptide.tolist(),
            #                              iPlausibleStruct.tolist(),
            #                              ims2by.cpu().numpy().tolist(),
            #                              ims2BY.cpu().numpy().tolist(),
            #                              icos,
            #                              self.id_list[self.nj]]
            # import ipdb
            # ipdb.set_trace()
            # print(self.repsequence)
            self.nj+=1 
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        data=pd.DataFrame(self.listcos)
        data.columns=[self.args.ms2_method]
        # import ipdb
        # ipdb.set_trace()
        mediancos=np.median(self.listcos)
        if mediancos>self.bestanswer:
            data.to_csv(self.savename+"similaritylist.csv",index=False)
            print(f"file saved {self.savename} similaritylist.csv")
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'medianmetric':round(mediancos,6),
                            'nan number':self.nan,
                           'averagemetric': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestmetric':round(self.bestcos, 6),
                           "metric":self.args.ms2_method
                           }
        self.repsequence.to_csv(self.savename+"_byBY_result.csv",index=False)
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

# --------------------------- rt ---------------------#
#rt的metric和loss是不是都要改一下，CS感觉不行
class CossimilarityMetricfortest_outputrt(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """
    def __init__(self, savename,pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred="predirt", target="irt", seq_len=seq_len,sequence=sequence,charge=charge,
        decoration=decoration,_id=_id)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.savename=savename if savename else ""
        self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','rt',"rt_target","id"])
        self.numcol=num_col
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """

        # ipdb.set_trace()
        if pred.dim() == 0:
            pred=pred.unsqueeze(0)
        N=pred.size(0) #batch size
        # N=pred.numel()
        # ipdb.set_trace()
        # if N!=128:
        #     ipdb.set_trace()
        # import ipdb
        # pred
        L=sequence.shape[1]
        # if seq_len is not None and target.dim() > 1:
        #     max_len = target.size(1)
        #     masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        # else:
        if rt_method=="cos":
            cos=CosineSimilarity(dim=0)
            s = torch.sum(cos(pred, target)).item()
        if rt_method=="delta":
            def delta(pred,target):
                return 1-torch.abs(pred-target)
            s = torch.sum(delta(pred,target)).item()
        if rt_method !="R2":
            self.cos += s
        self.total += N
        if rt_method =="cos":
            self.bestcos = max(self.bestcos, torch.max(cos(pred, target)).item())
        if rt_method =="delta":
            self.bestcos = max(self.bestcos, torch.max(delta(pred,target)).item())
            self.listcos += delta(pred,target).reshape(N, ).cpu().numpy().tolist()
            # ipdb.set_trace()
        # print(_id)
        # import ipdb
        # ipdb.set_trace()
        self.id_list+=_id.cpu().numpy().tolist()
        for i in range(N):
            il = seq_len[i]#il是序列真实长度
            isequence=sequence[i][:il]
            icharge=charge[i]
            idecoration=decoration[i][:il]
            iirt=pred[i]
            iirt_target=target[i]
            self.repsequence.loc[len(self.repsequence)] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         idecoration.cpu().numpy().tolist(),
                                         iirt.cpu().numpy().tolist(),
                                         iirt_target.cpu().numpy().tolist(),
                                         self.id_list[i]] #不是seld.nj，是i


    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        # data=pd.Series(self.listcos)
        # data.to_csv("Cossimilaritylist.csv",index=False)
        # self.repsequence.to_json(self.savename+"result.json")
        if rt_method =="cos":
            evaluate_result = {'mediancos':round(np.median(self.listcos),6),
                            'total number':len(self.repsequence),
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if rt_method =="delta":
            # ipdb.set_trace()
            evaluate_result = {'mediandelta':round(np.median(self.listcos),6),
                            'total number':len(self.repsequence),
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if rt_method=="R2":   
            r2 = r2_score(self.repsequence['rt'], self.repsequence['rt_target'])
            evaluate_result = {'r2':round(r2,6),
                            'total number':len(self.repsequence)
                           }
        if reset:
            r2=0
            self.cos = 0
            self.total = 0
            self.cos = 0
            self.listcos=[]
            self.id_list=[]
            self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','rt',"rt_target","id"])
        return evaluate_result


class CossimilarityMetricfortest_predrt(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """
    def __init__(self, savename,pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred="predirt", target="irt", seq_len=seq_len,sequence=sequence,charge=charge,
        decoration=decoration,_id=_id)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nj=0
        self.savename=savename if savename else ""
        self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','rt',"rt_target","id"])
        self.numcol=num_col
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,decoration=None,_id=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """

        N=pred.size(0) #batch size
        # import ipdb
        # ipdb.set_trace()
        L=sequence.shape[1]
        # if seq_len is not None and target.dim() > 1:
        #     max_len = target.size(1)
        #     masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        # else:
        if rt_method=="cos":
            cos=CosineSimilarity(dim=0)
            s = torch.sum(cos(pred, target)).item()
            self.cos += s
            self.bestcos = max(self.bestcos, torch.max(cos(pred, target)).item())
        if rt_method=="delta":
            def delta(pred,target):
                return 1-torch.abs(pred-target)
            s = torch.sum(delta(pred,target)).item()
            self.cos += s
            self.bestcos = max(self.bestcos, torch.max(delta(pred,target)).item())
            self.listcos += delta(pred,target).reshape(N, ).cpu().numpy().tolist()
        self.total += pred.size(0)
        self.id_list+=_id.cpu().numpy().tolist()
        for i in range(N):
            il = seq_len[i]#il是序列真实长度
            isequence=sequence[i][:il]
            icharge=charge[i]
            idecoration=decoration[i][:il]
            iirt=pred[i]
            iirt_target=target[i]
            self.repsequence.loc[len(self.repsequence)] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         idecoration.cpu().numpy().tolist(),
                                         iirt.cpu().numpy().tolist(),
                                         iirt_target.cpu().numpy().tolist(),
                                         self.id_list[len(self.repsequence)]] #不是seld.nj，是i

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        # data=pd.Series(self.listcos)
        # data.to_csv("Cossimilaritylist.csv",index=False)
        # import ipdb
        # ipdb.set_trace()
        repsequence_output=pd.DataFrame(self.repsequence)
        repsequence_output.to_csv(self.savename+"rtresult.csv")
        print(f"file saved {self.savename} rtresult.csv")
        if rt_method =="cos":
            evaluate_result = {'mediancos':round(np.median(self.listcos),6),
                            'total number':len(self.repsequence),
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if rt_method =="delta":
            # ipdb.set_trace()
            evaluate_result = {'mediandelta':round(np.median(self.listcos),6),
                            'total number':len(self.repsequence),
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if rt_method=="R2":   
            r2 = r2_score(self.repsequence['rt'], self.repsequence['rt_target'])
            evaluate_result = {'r2':round(r2,6),
                            'total number':len(self.repsequence)
                           }
        if reset:
            r2=0
            self.cos = 0
            self.total = 0
            self.cos = 0
            self.listcos=[]
            self.id_list=[]
            self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','rt',"rt_target","id"])
        return evaluate_result
    
# --------------------------- not in use ---------------------#    
class PearsonCCMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target)
        self.prelist=[]
        self.targetlist=[]

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        self.prelist+=pred.cpu().numpy().tolist()
        self.targetlist+=target.cpu().numpy().tolist()



    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        cos=CosineSimilarity(dim=0)
        MAE=-np.mean(np.abs(np.array(self.prelist)-np.array(self.targetlist)))
        Pprelist=self.prelist-np.mean(self.prelist)
        Ptargetlist=self.targetlist-np.mean(self.targetlist)
        PCC=cos(torch.Tensor(Pprelist),torch.Tensor(Ptargetlist))
        PCC=PCC.item()
        evaluate_result = {"meanl1loss":round(MAE,6),
                           'PCC':round(PCC,6),

                           }
        if reset:
            self.prelist = []
            self.targetlist=[]
        return evaluate_result
class PearsonCCMetricfortest(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target)
        self.prelist=[]
        self.targetlist=[]

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        self.prelist+=pred.cpu().numpy().tolist()
        self.targetlist+=target.cpu().numpy().tolist()



    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        cos=CosineSimilarity(dim=0)
        MAE=np.mean(np.abs(np.array(self.prelist)-np.array(self.targetlist)))
        Pprelist=self.prelist-np.mean(self.prelist)
        Ptargetlist=self.targetlist-np.mean(self.targetlist)
        PCC=cos(torch.Tensor(Pprelist),torch.Tensor(Ptargetlist))
        PCC=PCC.item()
        outdata=pd.DataFrame(columns=["pred_irt","exp_irt"])
        outdata["pred_irt"]=self.prelist
        outdata["exp_irt"]=self.targetlist
        outdata.to_csv("irt_pred_experiment.csv",index=False)
        evaluate_result = {"meanl1loss":round(MAE,6),
                           'PCC':round(PCC,6),

                           }
        if reset:
            self.prelist = []
            self.targetlist=[]
        return evaluate_result
# --------------------------- position embedding---------------------#
class PositionEmbedding(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,emb_size,maxlength):
        super().__init__()
        pe=torch.arange(0,maxlength)
        pe.requires_grad=False
        pe = pe.unsqueeze(0)
        self.embedding=nn.Embedding(maxlength,emb_size)
        self.register_buffer('pe', pe)#1LE
    def forward(self,x,device):
        pe=self.embedding(self.pe[:,:x.size(1)])
        return pe
import math
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# --------------------------- model--------------------#
class deepdiaModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.num_col=int(num_col)
        self.edim=embed_dim
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(2,embed_dim)#只有两种磷酸化情况
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,phos=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.LongTensor(range(L))
        slengths=ll.expand(N,L)
        slengths=slengths.to(device)
        sequence=peptide_tokens

        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(slengths)#NLE

        if phos:
            assert sequence.size(0) == phos.size(0)
            phos_embed = self.phos_embedding(phos)
            ninput=pos_embed+a_embed+phos_embed#NLE
        else:
            ninput = pos_embed + a_embed   #NLE
        key_padding_mask=attentionmask(peptide_length-1)
        ninput=self.activation(self.conv(ninput.permute(0,2,1)))#NE(L-1)
        output =self.transformer(ninput.permute(2,0,1),src_key_padding_mask=key_padding_mask)#(L-1)NE
        outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        outputms=self.dropout(self.mslinear(output))#(L-1)*N*12
        outputms=self.activation(outputms)
        outputms=outputms.permute(1,0,2).reshape(N,-1)#N*((L-1)*12)
        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(torch.sum(outputms))
        return {'pred':outputms}
class _2deepdiaModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=PositionEmbedding(embed_dim,maxlength)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)

        sequence=peptide_tokens
        device=peptide_tokens.device
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(peptide_tokens,device)#NLE


        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE

        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24
        outputms=self.activation(outputms)
        outputms=outputms.reshape(N,-1)#N*((L-1)*24)
        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(torch.sum(outputms))
        return {'pred':outputms}

# --------------------------- charge embedding --------------------#
class _2deepchargeModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(4,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE
        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*24)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}
# --------------------------- irt --------------------#
class _2deepchargeModelirt_ll(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear1=nn.Linear(embed_dim,256)
        self.rtlinear2=nn.Linear(256,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E
        output = self.activation(self.rtlinear1(output))
        outputrt=self.activation(self.rtlinear2(output).squeeze())#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}

# --------------------------- sigmoid --------------------#
class _2deepchargeModelirt_ll_sigmoid(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear1=nn.Linear(embed_dim,256)
        self.rtlinear2=nn.Linear(256,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E
        output = self.activation(self.rtlinear1(output))
        outputrt=torch.sigmoid(self.rtlinear2(output).squeeze())#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}

# --------------------------- clsembedding --------------------#
class _2deepchargeModelirt_cls(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.cls_embedding=nn.Embedding(2,embed_dim)##
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L+1,device=device).unsqueeze(0)##L+1
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        cls_embed=self.cls_embedding(torch.ones(N,device=device,dtype=int)).unsqueeze(1)#N*1*E

        pos_embed=self.pos_embedding(ll)#1(L+1)E

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)#NLE
        ninput=a_embed+phos_embed#NLE
        ninput=torch.cat([cls_embed,ninput],dim=1)+pos_embed#N(L+1)E
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length+1)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L+1)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
#######################
class _2deepchargeModelirt(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
##################################################################取0位置linear
class _2deepchargeModelirt_zero(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(4,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}

######完整版模型#############################
class _2deepchargeModelirt_zero_all(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}


class _2deepchargeModelms2_all(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration,pnumber):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE
        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}

########################contrast training for decoy
class _2deepchargeModelms2_all_contrast(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration,pnumber,false_samples=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #false_samples:N*F(number of decoy of every sample)*L 这里面只存了phos的数据.
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE

        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)
        if false_samples is None:
            return {'pred': outputms, 'sequence': sequence, 'charge': charge, "decoration": decoration,
                    "seq_len": peptide_length,
                    'pnumber': pnumber}
        # print(outputms)
        # print(torch.sum(outputms))
        else:
            false_phos_embed = self.phos_embedding(false_samples)
            false_ninput = pos_embed + a_embed + charge_embed  ##NLE
            false_ninput=false_ninput.unsqueeze(1)
            # import ipdb
            # ipdb.set_trace()
            false_ninput = false_ninput + false_phos_embed
            F = false_ninput.size(1)
            false_ninput = false_ninput.reshape(N * F, L, -1)
            false_peplen = peptide_length.expand(F, N).T.reshape(N * F)
            false_key_padding_mask = attentionmask(false_peplen)

            false_output = self.transformer(false_ninput.permute(1, 0, 2),
                                            src_key_padding_mask=false_key_padding_mask)  # (L)(N*F)E

            false_output = self.activation(self.conv(false_output.permute(1, 2, 0)))  # (N*F)E(L-1)
            false_output = false_output.permute(0, 2, 1)  # (N*F)*(L-1)*E
            false_outputms = self.dropout(self.mslinear(false_output))  # (N*F)*(L-1)*24

            false_outputms = self.activation(false_outputms)
            false_outputms=false_outputms.reshape(N*F,-1)   # (N*F)*((L-1)*num_col)
            false_seq_len = ((peptide_length - 1) * self.num_col).expand(F, N).T.reshape(N * F)

            false_masks = seq_len_to_mask(seq_len=false_seq_len)  ##加上mask

            false_outputms = false_outputms.masked_fill(false_masks.eq(False), 0)
            false_outputms = false_outputms.reshape(N, F, -1)  # N*F*((L-1)*num_col)
            return {'pred': outputms, 'sequence': sequence, 'charge': charge, "decoration": decoration,
                    "seq_len": peptide_length,
                    'pnumber': pnumber, "false_outputms": false_outputms}
