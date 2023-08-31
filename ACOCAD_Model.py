import tensorflow as tf
import keras
from keras import backend as K,Input
from keras.layers import Lambda
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Activation, LeakyReLU, \
    LSTM, Embedding,  MultiHeadAttention,Softmax
from constracted_embedding_matrix import *

class ACOCAD_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d = 512
        self.question_length = 14
        self.num_heads = 8
        self.deepmodule = 6
        self.num_regions,self.regions_dimension=36,2048
        self.num_PossibleAnswer=3129
        self.V_Frcnn = Input(shape=(self.num_regions, self.regions_dimension), name='in1')
        self.Q = Input(shape=[self.question_length,], name='in2')
        self.Q_USE = Input(shape=(512), name='in3')
        self.Qe = K.expand_dims(self.Q_USE,axis=1)

    def Embedding_layar_glove(self):
        embedding_matrix, word_index = emd()
        embedding = Embedding(len(word_index) + 1,  # number of unique tokens
                              300,  # number of features
                              embeddings_initializer=Constant(embedding_matrix),  # initialize
                              input_length=self.question_length,
                              trainable=True)(self.Q)
        return embedding

    def Feedforwardinglayer(self, in_FF):
        f1 = Dense(4 * self.d, activation='relu')(in_FF)
        f_drop = Dropout(.2)(f1)
        f2 = Dense(self.d)(f_drop)
        return f2

    def Lstm_NN(self):
        E = self.Embedding_layar_glove()
        E_drop = Dropout(.4)(E)
        Q_lstm = LSTM(self.question_length, input_shape=(self.question_length, 300), return_sequences=True)(E_drop)

        return Q_lstm

    def CAGA(self,Q_lstm, Q_e):
        Multi_NN, ___ = MultiHeadAttention(num_heads=self.num_heads, key_dim=2)(query=Q_lstm,value=Q_e,return_attention_scores=True)
        A_CAGA = tf.keras.layers.Add()([Multi_NN, Q_lstm])
        N_CAGA = tf.keras.layers.LayerNormalization(axis=1)(A_CAGA)
        return N_CAGA

    def SA(self, input_SA):
        print(input_SA)
        Multi_NN, ___ = MultiHeadAttention(num_heads=self.num_heads, key_dim=2)(query=input_SA, value=input_SA,return_attention_scores=True)

        A_SA = tf.keras.layers.Add()([Multi_NN, input_SA])
        N_SA = tf.keras.layers.LayerNormalization(axis=1)(A_SA)
        SA_F = self.Feedforwardinglayer(N_SA)
        return SA_F

    def Maxpooling(self, in_Pooling):
        out_p = Lambda(lambda xx: keras.backend.max(xx, keepdims=True, axis=1))(in_Pooling)
        A_P = tf.keras.layers.Add()([out_p, self.Q_USE])
        N_P = tf.keras.layers.LayerNormalization(axis=1)(A_P)
        return N_P

    def QLGA(self, input_QU,input_v):
        input_v = Dense(self.d)(input_v)
        Multi_NN, ___ = MultiHeadAttention(num_heads=self.num_heads, key_dim=2, value_dim=2)(query=input_v,value=input_QU,return_attention_scores=True)
        A_QLGA = tf.keras.layers.Add()([Multi_NN, input_v])
        N_QLGA = tf.keras.layers.LayerNormalization(axis=1)(A_QLGA)
        D_QLGA = Dense(self.d)(N_QLGA)
        return D_QLGA

    def WLGA(self, input_QLGA, input_w):
        Multi_NN, ___ = MultiHeadAttention(num_heads=self.num_heads, key_dim=2, value_dim=2)(query=input_QLGA,value=input_w,return_attention_scores=True)
        A_WLGA = tf.keras.layers.Add()([Multi_NN, input_QLGA])
        N_WLGA = tf.keras.layers.LayerNormalization(axis=1)(A_WLGA)
        F_WLGA = self.Feedforwardinglayer(N_WLGA)
        return F_WLGA

    def DMF(self, in_words, in_regions):
        D_words = Dense(self.d)(in_words)
        D_regions = Dense(self.d)(in_regions)

        #### Attention map
        ATT_map = K.batch_dot(D_words,D_regions,axes=2)

        #### dependent attention mechanisem
        row_side = K.sum(ATT_map, keepdims=False, axis=1) #36
        column_side = K.sum(ATT_map, keepdims=False, axis=2) #14

        soft_im =  Softmax(axis=1)(row_side)
        soft_word = Softmax(axis=1)(column_side)


        im_att = keras.layers.Reshape((self.num_regions, 1))(soft_im)
        q_att = keras.layers.Reshape((self.question_length, 1))(soft_word)

        im_vectors = keras.layers.multiply([in_regions, im_att])
        q_vectors = keras.layers.multiply([in_words, q_att])

        Vr = tf.keras.layers.Add()([im_vectors, in_regions])
        Qw = tf.keras.layers.Add()([q_vectors, in_words])

        ##### independent attention mechanisem
        D_Vr= Dense(self.d, activation='relu')(Vr)
        D_Qw= Dense(self.d, activation='relu')(Qw)

        row_side = K.sum(D_Vr, keepdims=True, axis=2)  # 36
        column_side = K.sum(D_Qw, keepdims=True, axis=2)  # 14

        soft_im = Softmax(axis=1)(row_side)
        soft_word = Softmax(axis=1)(column_side)

        im_vectors = keras.layers.multiply([Vr, soft_im])
        q_vectors = keras.layers.multiply([Qw, soft_word])

        Vr2 = tf.keras.layers.Add()([im_vectors, in_regions])
        Qw2 = tf.keras.layers.Add()([q_vectors, in_words])

        F_regions_DMF = K.sum(Vr2, keepdims=False, axis=1)
        F_words_DMF = K.sum(Qw2, keepdims=False, axis=1)
        return F_regions_DMF, F_words_DMF

    def fusion(self, in_f1, in_f2):
        final_merge = keras.layers.multiply([in_f1, in_f2])
        return final_merge

    def classification_layers(self, in_CL):
        CL_d = Dense(self.d)(in_CL)
        CL_d = LeakyReLU(.2)(CL_d)
        CL_d = Dropout(.4)(CL_d)
        out_L = Dense(self.num_PossibleAnswer)(CL_d)
        Final_answer = Activation('sigmoid')(out_L)
        return Final_answer



    def main(self):
        for i in range(self.deepmodule):
            if i==0:
                GQ = self.CAGA(self.Lstm_NN(),self.Qe)
                QL = self.SA(GQ)
                QU = self.Maxpooling(QL)
            else:
                GQ = self.CAGA(QL, QU)
                QL = self.SA(GQ)
                QU = self.Maxpooling(QL)

        for j in range(self.deepmodule):
            if j==0:
                G1V = self.QLGA(QU,self.V_Frcnn)
                G2V = self.WLGA(G1V, QL)
                V2 = self.SA(G2V)
            else:
                G1V = self.QLGA(QU, V2)
                G2V = self.WLGA(G1V, QL)
                V2 = self.SA(G2V)

        F_words, F_regions = self.DMF(QL, V2)
        Fusion_layer = self.fusion(F_words, F_regions)
        outputlayer=self.classification_layers(Fusion_layer)

        VQA_model = Model(inputs=[self.V_Frcnn, self.Q, self.Q_USE], outputs=outputlayer)
        VQA_model.summary()
        return VQA_model

# model=ACOCAD_model()
# VQA_model=model.main()
# print(VQA_model.summary())
