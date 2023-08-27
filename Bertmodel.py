from transformers.models.bert.modeling_bert import (BertEmbeddings, BertModel,
BertForSequenceClassification,BertPreTrainedModel,BertEncoder,BertPooler)
from transformers.models.roberta.modeling_roberta import RobertaModel,RobertaEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
import GNN_global_representation
import GNN_edge_regression
from utils import *

class Acid_BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.charge_embedding=nn.Embedding(10,config.hidden_size,padding_idx=0)
        self.a_embedding=nn.Embedding(30,config.hidden_size,padding_idx=0)
        self.phos_embedding=nn.Embedding(10,config.hidden_size)#修饰三种加上padding###全部调大了，其他的修饰也在这里
        print(f"GNN_global_ablation {GNN_global_ablation}!!!")
        if GNN_global_ablation=="GIN":
            self.gly_embedding=GNN_global_representation.GIN(20, GNN_global_hidden_dim, config.hidden_size,init_eps=0)
            #16也可以改
        if GNN_global_ablation=="GCN":
            self.gly_embedding=GNN_global_representation.GCN(20, GNN_global_hidden_dim, config.hidden_size)
            # print("input GCN model")
        if GNN_global_ablation=="GAT":
            self.gly_embedding=GNN_global_representation.GAT(20, GNN_global_hidden_dim, config.hidden_size,num_heads=4)
        if GNN_global_ablation=="Nogly":
            pass
        if GNN_global_ablation not in ["GIN","GCN","GAT","Nogly"]:
            raise NameError
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


    def forward(
        self, peptide_tokens=None, position_ids=None,decoration=None,charge=None,batched_graph=None, inputs_embeds=None, past_key_values_length=0
    ):  
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))
        feat=batched_graph.ndata["attr"]
        if GNN_global_ablation=="GIN" or GNN_global_ablation =="GCN" or GNN_global_ablation =="GAT":
            gly_embedding=self.gly_embedding(batched_graph,feat)
            gly_embedding=gly_embedding.unsqueeze(1).expand(N,L,gly_embedding.size(1))*((decoration==5).unsqueeze(-1).expand(
    N,L,gly_embedding.size(1)))
        if GNN_global_ablation=="Nogly":
            pass

        assert sequence.size(0) == decoration.size(0)
        # ipdb.set_trace()
        #decoration传入了decoration_ids，包括了decoration_ACE的信息，见preprocess.py
        phos_embed = self.phos_embedding(decoration)
        
        if loc==False:
            phos_embed = self.phos_embedding(decoration-5*(decoration==5).to(int))

        if peptide_tokens is not None:
            input_shape = peptide_tokens.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.a_embedding(peptide_tokens)
        if GNN_global_ablation !="Nogly":
            embeddings = inputs_embeds+phos_embed+charge_embed+gly_embedding
        if GNN_global_ablation =="Nogly":
            embeddings = inputs_embeds+phos_embed+charge_embed

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# ----------------------- byBY ------------------------------#
class ModelbyBYms2_bert(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=num_col
        self.mslinear=nn.Linear(config.hidden_size,num_col)
        print(f"GNN_edge_ablation {GNN_edge_ablation}!!!")
        if GNN_edge_ablation=="GIN":
            self.BY_pred=GNN_edge_regression.GIN(20, GNN_edge_hidden_dim, config.hidden_size,init_eps=0)
        if GNN_edge_ablation=="GCN":
            self.BY_pred=GNN_edge_regression.GCN(20, GNN_edge_hidden_dim, config.hidden_size)
        if GNN_edge_ablation=="GAT":
            self.BY_pred=GNN_edge_regression.GAT(20, GNN_edge_hidden_dim, config.hidden_size,num_heads=4)
        self.peptide_rep_linear=nn.Linear(config.hidden_size,GNN_edge_hidden_dim)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,
                charge,decoration,decoration_ids,_id,decoration_ACE,PlausibleStruct,
                peptide,graph_edges,ions_by_p,ions_BY_p,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        # import ipdb
        # ipdb.set_trace()
        key_padding_mask=seq_len_to_mask(peptide_length*2)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        # past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        batched_graph=PlausibleStruct
        # import sys
        # sys.path.append('..')
        from masses import glyco_process
        import dgl
        graph_list=[]
        for i in batched_graph:
            _,_,Struct_tokens,nodef=glyco_process(i)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            grapbatched_graphh=dgl.graph(Struct_tokens)
            grapbatched_graphh.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            graph_list.append(grapbatched_graphh)
        batched_graph = dgl.batch(graph_list).to(device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, batched_graph=batched_graph,
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        #by
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col
        outputms=outputms[:,:,:self.num_col]
        outputms=self.activation(outputms)
        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)
        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)
        # print(outputms)
        # print(torch.sum(outputms))

        #BY
        batched_graph = dgl.batch(graph_list).to(device)
        feat = batched_graph.ndata["attr"]
        peptide_ind=torch.nonzero(feat==0).squeeze()
        peptide_rep=self.peptide_rep_linear(sequence_output[:,0,:])
        BY_pred=self.BY_pred(batched_graph,feat,peptide_rep,peptide_ind) #利用global_edge_regression对于图以及节点坐标得到边的碎裂可能性

        pred_BY = torch.split(BY_pred, graph_edges.tolist())
        ##padding BY
        contents=[BY.reshape(-1,1).squeeze() for BY in pred_BY]
        max_len = max(map(len, contents))
        tensor = torch.full((len(contents), max_len), fill_value=0,dtype=torch.float)
        for i, content_i in enumerate(contents):
            tensor[i, :len(content_i)] = content_i
        pred_BY=tensor.to(device)
        concatbyBY=torch.cat([outputms,pred_BY],dim=-1)#128,512 batchsize(padding_by_size*padding_BY_size)

        #target
        target_BY = torch.split(ions_BY_p, graph_edges.tolist())
        ##padding BY
        contents=[BY.reshape(-1,1).squeeze() for BY in target_BY]
        max_len = max(map(len, contents))
        tensor = torch.full((len(contents), max_len), fill_value=0,dtype=torch.float)
        for i, content_i in enumerate(contents):
            tensor[i, :len(content_i)] = content_i
        target_BY=tensor.to(device)
        target_byBY=torch.cat([ions_by_p,target_BY],dim=-1)#128,512 batchsize(padding_by_size*padding_BY_size)
        # import ipdb
        # ipdb.set_trace()
        return {'pred_by':outputms,'pred_BY':BY_pred,"pred":concatbyBY,
                'target_by':ions_by_p,'target_BY':ions_BY_p,"target":target_byBY,
                'sequence':peptide_tokens,'charge':charge,
                "decoration":decoration,"seq_len":peptide_length,"_id":_id,
                "PlausibleStruct":PlausibleStruct,"peptide":peptide,"graph_edges":graph_edges}


# ----------------------- by ------------------------------#
class _2deepchargeModelms2_bert(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=num_col
        self.mslinear=nn.Linear(config.hidden_size,num_col)
        # model_ablation="DeepFLR" #DeepFLR
        # if model_ablation=="DeepFLR":
        #     self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,_id,decoration_ACE,PlausibleStruct,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        # import ipdb
        # ipdb.set_trace()
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        # past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        batched_graph=PlausibleStruct
        # import sys
        # sys.path.append('..')
        from masses import glyco_process
        import dgl
        graph_list=[]
        for i in batched_graph:
            _,_,Struct_tokens,nodef=glyco_process(i)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            grapbatched_graphh=dgl.graph(Struct_tokens)
            grapbatched_graphh.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            graph_list.append(grapbatched_graphh)
        batched_graph = dgl.batch(graph_list).to(device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, batched_graph=batched_graph,
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col
        ##
        outputms=outputms[:,:,:self.num_col]
        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,"_id":_id}


# ----------------------- rt ------------------------------#
class _2deepchargeModelms2_bert_irt(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=num_col #有多少碎片
        # self.mslinear=nn.Linear(config.hidden_size,36)#这个应该不用了吧
        self.pooler = BertPooler(config)
        self.irtlinear=nn.Linear(config.hidden_size,1)
        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,_id,
    decoration_ACE,PlausibleStruct,return_dict=None,head_mask=None):
    #增加了_id,decoration_ACE,PlausibleStruct
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        batched_graph=PlausibleStruct
        import sys
        sys.path.append('..')
        from masses import glyco_process
        import dgl
        graph_list=[]
        for i in batched_graph:
            _,_,Struct_tokens,nodef=glyco_process(i)
            node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}
            nodef="P"+nodef
            Struct_feature=[node2idx[i] for i in nodef]
            grapbatched_graphh=dgl.graph(Struct_tokens)
            grapbatched_graphh.ndata["attr"]=torch.Tensor(Struct_feature).to(int)
            graph_list.append(grapbatched_graphh)
        batched_graph = dgl.batch(graph_list).to(device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, batched_graph=batched_graph,
        position_ids=None,decoration=decoration_ids,charge=charge, inputs_embeds=None, 
        past_key_values_length=0
        )    
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        predirt=self.irtlinear(pooled_output).squeeze()##
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        #outputms一定要有吗，弄一下尺寸，现在loss是nan
        # outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col
        # outputms=outputms[:,:,:self.num_col]
        # outputms=self.activation(outputms)
        # outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)
        # masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        # outputms=outputms.masked_fill(masks.eq(False), 0)
        return {'sequence':peptide_tokens,'charge':charge,"decoration":decoration,
        "seq_len":peptide_length,"_id":_id,"predirt":predirt}



# ----------------------- models not in use ------------------------------#              
class _2deepchargeModelms2_roberta(RobertaModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col
        outputms=self.activation(outputms)
        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_bert_ss(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        outputss=self.sslinear(output)
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_bert_ss_contrast(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        ############no_grad_no_linear_contrast, simsiam
        encoder_outputs_ss = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output_ss=encoder_outputs_ss[0]
        output_nograd=sequence_output_ss[:,2:-1:2,:].detach()
        outputss=self.sslinear(output)
        ####对比outputss与output_nograd.size:B*(L-1)*E
        contrastmask=seq_len_to_mask(seq_len=(peptide_length - 1) *E )
        outputss_masked=outputss.reshape(batch_size,-1).masked_fill(contrastmask.eq(False), 0)
        output_nograd_masked=output_nograd.reshape(batch_size,-1).masked_fill(contrastmask.eq(False), 0)###B*((L-1)*E)
        p=F.normalize(outputss_masked,p=2,dim=1)
        z=F.normalize(output_nograd_masked,p=2,dim=1)
        sscontrastloss=-(p*z).sum(dim=1).mean()
        
        
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {"sscontrastloss":sscontrastloss,'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_roberta_ss(RobertaModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        outputss=self.sslinear(output)
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
