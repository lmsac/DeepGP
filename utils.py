import fastNLP
import wandb
GNN_edge_ablation="GIN" #GIN
GNN_edge_hidden_dim=128 #64,128
GNN_edge_decoder_type="mlp"
GNN_edge_num_layers=7

GNN_global_ablation="GCN"  #GCN
GNN_global_hidden_dim=32
GNN_global_num_layers=7  #7

loc=True
rt_method="R2" #R2,cos,delta


acid_size=22
N_epochs=150
vocab_save=False
vocab=fastNLP.Vocabulary().load("AA_vocab")
BATCH_SIZE=256
# device="cuda:0" if torch.cuda.is_available () else "cpu"
# device="cpu"
num_col=24
seed=101
dropout=0.2
embed_size=512
nhead=8
# num_layers=6
# maxlength=60
# finetune_epoch=10


class WandbCallback(fastNLP.Callback):
    r"""
    
    """
    def __init__(self,project,name,config:dict):
        r"""
        
        :param str name: project名字
        :param dict config: 模型超参
        :
        """
        super().__init__()
        self.project = project
        self.name=name
        self.config=config
    def on_train_begin(self):
        wandb.init(
      # Set entity to specify your username or team name
      # ex: entity="carey",
      # Set the project where this run will be logged
      project=self.project,
      name=self.name, 
      # Track hyperparameters and run metadata
      config=self.config)
    def on_train_end(self):
        wandb.finish()
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        wandb.log(eval_result)
    def on_backward_begin(self,loss):
        wandb.log({"loss":loss})