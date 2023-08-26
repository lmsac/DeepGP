from fastNLP import LossBase
import torch.nn.functional as F
import torch
class MSELoss_byBY(LossBase):
    r"""
    MSE损失函数

    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` >`target`
    :param str reduction: 支持'mean'，'sum'和'none'.

    """

    def __init__(self, pred_by=None,pred_BY=None,target_by=None,target_BY=None, reduction='mean'):
        super(MSELoss_byBY, self).__init__()
        self._init_param_map(pred_by=pred_by,pred_BY=pred_BY,target_by=target_by,target_BY=target_BY)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def get_loss(self, pred_by,pred_BY, target_by,target_BY):
        byloss=F.mse_loss(input=pred_by.float(), target=target_by.float(), reduction=self.reduction)
        BYloss=F.mse_loss(input=pred_BY.float(), target=target_BY.float(), reduction=self.reduction)
        # byloss=F.mse_loss(input=torch.log2(pred_by.float()+1), target=torch.log2(target_by.float()+1), reduction=self.reduction)
        # import ipdb
        # ipdb.set_trace()
        # print("loss ratio",BYloss/byloss)
        loss=50*byloss+BYloss
        # loss=byloss+0.02*BYloss
        return loss
