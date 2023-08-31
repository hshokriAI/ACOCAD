from model_mix_gru_universal_new import *
from DataGenerator import DataGenerator_train, DataGenerator_test
from keras.models import load_model
import tensorflow as tf
import numpy as np
import keras
from keras import optimizers
from ACOCAD_Model import ACOCAD_model


class VQA:
    def __init__(self, questions_id, images_id, answers, one_hot_dic_answer, emd_vec, batchsize, question_all,
                 USE_data):
        self.questions_id = questions_id
        self.images_id = images_id
        self.answers = answers
        self.one_hot_dic_answer = one_hot_dic_answer
        self.emd_vec = emd_vec
        self.batch_size = batchsize
        self.question_all = question_all
        self.USE_data = USE_data

    def Train_model(self):
        model_checkpoint = keras.callbacks.ModelCheckpoint(save_best_only=True)

        ### create new model
        ACOCAD_instance = ACOCAD_model()
        model_nn = ACOCAD_instance.main()
        model_nn.compile(loss='binary_crossentropy', optimizer=optimizers.adamax(clipnorm=.25))

        training_generator = DataGenerator_train(questions_id=self.questions_id, images_id=self.images_id,
                                                 answers=self.answers,
                                                 one_hot_dic_answer=self.one_hot_dic_answer, emd_vec=self.emd_vec,
                                                 batch_size=self.batchsize,
                                                 shuffle=True,
                                                 questions=self.question_all,
                                                 USE_data=self.USE)

        model_nn.fit_generator(generator=training_generator,
                               use_multiprocessing=False, epochs=60,
                               verbose=1, max_queue_size=100, callbacks=[model_checkpoint])

        return model_nn

    def Test_model(self):
        model = load_model(
            'trianed_model.hdf5',
            custom_objects={'tf': tf, 'keras': keras})

        validation_generator = DataGenerator_test(questions_id=self.questions_id, images_id=self.images_id,
                                                  answers=self.answers,
                                                  one_hot_dic_answer=self.one_hot_dic_answer, emd_vec=self.emd_vec,
                                                  batch_size=self.batchsize,
                                                  shuffle=False,
                                                  questions=self.question_all,
                                                  USE_data=self.USE)

        y = model.predict_generator(validation_generator, max_queue_size=100, verbose=1)
        condidate_answers = VQA.target_answers_test(self)

        y_sort = np.sort(y, axis=1)
        y_sort_index = np.argsort(y, axis=1)

        self.y_final = []
        self.y_final_value = []
        for index in range(len(y_sort_index)):
            self.y_final.append(y_sort_index[index][-1])
            self.y_final_value.append(y_sort[index][-1])

        predict_answers = [[self.switch_one_hot_dic[num[0]]] for num in self.y_final]

        return VQA.accuracy(self, predict_answers, condidate_answers)


if __name__ == '__main__':
    VQA_instanse = VQA(questions_id, images_id, answers, one_hot_dic_answer, emd_vec, batchsize, question_all,
             USE_data)
    VQA_instanse.Train_model()
    VQA_instanse.Test_model()
