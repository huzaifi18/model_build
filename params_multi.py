# Data
pth = 'model_build/data/NASA'
seq_len = 5 # time steps or history window size
hop = 1 # overlap between steps
k = 3 # kfold cross validation
random = 1
sample = 10

# Model
# Change model name by desired (i.e. singleLSTM, singleGRU, singleMGU, GatedConv1D, stackedLSTM, stackedGRU)
from model import *
# model = Transformer
#model = T2V_LSTM
model = singleLSTM

# Training hyperparameter for LSTM
lr = 0.001 # default Adam optim
batch_size = 50
epochs = 100
hidden_dim = 32
dropout_rate = 0.

# Training params for Transformer
time_embed = False # True = using T2V, False = not using T2V
num_head = 3 # Coba 1 dan 3 head
num_block = 2
ff_dim = 64
dropout_enc = 0.

# Save Directory
test_data = 'B18' # change based on which test data is used in the current experiment
save_dir = 'model_build/saved_models_multi/'+str(model.__name__)+'_'+test_data
# save_dir = 'code_keras/saved_models_multi/'+str(model.__name__)+'_'+test_data+'_'+str(time_embed)+'_head'+str(num_head)
model_dir = str(model.__name__)+'_h'+str(hidden_dim)+'_bs'+str(batch_size)+ \
			'_ep'+str(epochs)+'_w'+str(seq_len)+'_Adam_lr'+str(lr)
# model_weight = save_dir+"/"+model_dir+"/weight"