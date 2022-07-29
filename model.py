import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Layer, TimeDistributed, MultiHeadAttention
from tensorflow.keras.layers import Dense, Flatten, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Activation, Convolution2D, ConvLSTM2D, Conv2D, Conv1D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Dropout, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2

class Time2Vec(Layer):
    def __init__(self, seq_len=None, **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        self.seq_len = seq_len

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'seq_len': self.seq_len,
        })
        return config

    def build(self, input_shape):
        self.wp = self.add_weight(name='wei_per',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
        self.bp = self.add_weight(name='bias_per',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
        # linear
        self.wl = self.add_weight(name='wei_lin',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
        self.bl = self.add_weight(name='bias_lin',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
        super(Time2Vec, self).build(input_shape)
        
    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:], axis=-1) # Convert (batch, seq_len, 31) to (batch, seq_len)
        time_linear = self.wl * x + self.bl
        time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
        time_periodic = tf.math.sin(tf.multiply(x, self.wp) + self.bp)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2)
        
def TransformerEncoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention part
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    res = LayerNormalization(epsilon=1e-6)(x + inputs) # add & norm

    x = Dense(ff_dim, activation='relu')(res)
    x = Dense(head_size)(x)
    x = Dropout(dropout)(x)
    return LayerNormalization(epsilon=1e-6)(x + res) # add & norm

def Transformer(input_shape, num_hid, time_steps, num_head, ff_dim, num_layers_enc,  
                time_embedding=True, dropout=0.):
    inputs = Input(shape=input_shape)
    num_hid = num_hid # feat_dim=31
    if time_embedding:
        num_hid += 2 #additional 2 for timevec linear & periodic
        tv = Time2Vec(time_steps)(inputs)
        x = Concatenate()([inputs, tv])
        for _ in range(num_layers_enc):
            x = TransformerEncoder(x, num_hid, num_head, ff_dim, dropout)
    else:
        x = inputs
        for _ in range(num_layers_enc):
            x = TransformerEncoder(x, num_hid, num_head, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format='channels_last')(x)
    x = Dense(32)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

def T2V_LSTM(input_shape, hidden_dim):
    model = Sequential()
    model.add(Time2Vec(1, input_shape=input_shape))
    # model.add(Time2Vec(input_shape[-1], input_shape=input_shape)) #input_shape[-1]=31
    model.add(LSTM(hidden_dim, activation = 'tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def singleLSTM(input_shape, hidden_dim):
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=input_shape, activation = 'tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    return model

# Single LSTM/GRU/MGU with Dense layers
# Input shape for LSTM/GRU/MGU: input_shape=(timesteps, feat_dim) 
# or batch_input_shape=(batch_size, timesteps, feat_dim), needed if stateful=True
#def singleLSTM(input_shape, hidden_dim):
    #model = Sequential()
    #model.add(LSTM(5, input_shape=input_shape, activation = None, return_sequences=False))
    #model.add(Dropout(0.5))
    #model.add(LeakyReLU())
    #model.add(LSTM(hidden_dim, activation='tanh', return_sequence=True))
    #model.add(LSTM(hidden_dim, activation='tanh', return_sequence=True))
    #model.add(LSTM(hidden_dim, activation='tanh', return_sequence=True))
    #model.add(LSTM(hidden_dim, return_sequences=False))
    #model.add(Dropout(0.5))
    #model.add(LeakyReLU())
    #model.add(Dense(32))
    #model.add(Dropout(0.05))
    #model.add(Dense(32))
    #model.add(Dropout(0.5))
    #model.add(LeakyReLU())
    #model.add(Dense(1))
    #return model
    
def singleGRU(input_shape, hidden_dim):
    model = Sequential()
    model.add(GRU(hidden_dim, input_shape=input_shape, activation='tanh', return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def singleMGU(input_shape, hidden_dim, model_name='basic_mgu'):
    print('Define network:', model_name)
    model = Sequential()
    if model_name == 'basic_mgu':
        mgu_basic = MGUBasicModel(implementation=1, units=hidden_dim,
                                activation='tanh', input_shape=input_shape)
        model.add(mgu_basic)
    model.add(Dense(32))
    model.add(Dense(1))
    return model

# Stacked LSTM/GRU
def stackedLSTM(input_shape, hidden_dim):
    model = Sequential()
    model.add(LSTM(hidden_dim[0], input_shape=input_shape, activation='tanh', dropout=0., return_sequences=True))
    model.add(LSTM(hidden_dim[1], activation='tanh', dropout=0.))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def stackedGRU(input_shape, hidden_dim):
    model = Sequential()
    model.add(GRU(hidden_dim[0], input_shape=input_shape, activation='tanh', dropout=0., return_sequences=True))
    model.add(GRU(hidden_dim[1], activation='tanh', dropout=0., return_sequences=True))
    model.add(GRU(hidden_dim[2], activation='tanh', dropout=0.))
    model.add(Dense(32))
    model.add(Dense(1))
    return model


# Single GatedCNN
def GatedConv1D(input_shape, nb_filter, nb_kernel, nb_stride, dropout_rate):
    # Linear
    input_ = Input(input_shape)
    lin_out = Conv1D(nb_filter, nb_kernel, strides=nb_stride, kernel_initializer='random_uniform', padding='same')(input_)
    lin_out = BatchNormalization()(lin_out)

    # Conv1d with sigmoid activation function
    sigmoid_out = Conv1D(nb_filter, nb_kernel, strides=nb_stride, kernel_initializer='random_uniform', padding='same')(input_)
    sigmoid_out = BatchNormalization()(sigmoid_out)
    sigmoid_out = Activation('sigmoid')(sigmoid_out)

    # Merged
    out = multiply([lin_out, sigmoid_out])
    # out = MaxPooling1D(2)(out)
    out = MaxPooling1D(1)(out)
    out = Flatten()(out)
    out = Dense(32)(out)
    out = BatchNormalization()(out)
    # out = Dropout(dropout_rate)(out)
    out = Dense(1)(out)

    model = Model(input_, out)
    return model
