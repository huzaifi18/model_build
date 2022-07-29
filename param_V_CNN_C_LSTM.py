# Data
pth = 'model_build/data/NASA'
seq_len_lstm = 5  # time steps or history window size
seq_len_cnn = 5
hop = 1  # overlap between steps
k = 3  # kfold cross validation
random = 1
sample = 10

lr = 0.001  # default Adam optim
batch_size = 50
epochs = 100

test_data = '_B07'  # change based on which test data is used in the current experiment
save_dir = 'saved/'
model_dir = "SC-CNN-LSTM_" + str(seq_len_cnn) + test_data