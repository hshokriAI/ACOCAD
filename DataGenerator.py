import keras
import os
from w2v import *
from pardazesh import *
from sklearn import preprocessing


class DataGenerator_train(keras.utils.Sequence):
    'Generates data for model'

    def __init__(self, questions_id, images_id, answers, one_hot_dic, batch_size, shuffle, questions, emd_vec, USE):
        self.batch_size = batch_size
        self.questions_id = questions_id
        self.images_id = images_id
        self.answers = answers
        self.one_hot_dic = one_hot_dic
        self.shuffle = shuffle
        self.emd_vec = emd_vec
        self.on_epoch_end()
        self.question_all = questions
        self.universal_file = USE
        self.question_length = 14
        self.num_regions, self.regions_dimension = 36, 2048
        self.num_PossibleAnswer = 3129

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.questions_id) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.questions_id[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # print(X[0].shape,X[1].shape,y.shape)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.questions_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_question_lstm = np.zeros((self.batch_size, self.question_length,))
        X_question_USE = np.zeros((self.batch_size, 512))
        X_image = np.zeros((self.batch_size, self.num_regions, self.regions_dimension))
        y = np.zeros((self.batch_size, self.num_PossibleAnswer))

        for count, ID in enumerate(list_IDs_temp):
            directory = os.getcwd() + self.universal_file + str(ID) + '.npy'
            X_question_USE[count,] = np.load(directory)
            q = self.question_all[ID].lower()
            VC = self.emd_vec[ID]
            im_id = str(self.images_id[str(ID)])
            directory = os.getcwd() + 'image_directory/' + im_id + '.npy'
            im_norm = preprocessing.normalize(np.load(directory), norm='l2')

            X_image[count,] = im_norm
            X_question_lstm[count,] = VC

            answers_group = self.answers[ID]

            for k in answers_group:
                pre_proc_answer = preprocess_answer(k['answer'])
                try:
                    if y[count, self.one_hot_dic[pre_proc_answer]] < 0.96:
                        y[count, self.one_hot_dic[pre_proc_answer]] += 0.33

                except:
                    pass

            X = [X_image, X_question_lstm, X_question_USE]

        return X, y
