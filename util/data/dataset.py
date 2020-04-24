import torch.utils.data as data
import numpy as np
import random

from util.data.create_database import generate_sparse_matrix_of_genre, get_genre_collection, generate_sparse_matrix_from_multiple_genres


def get_dataset(genreA, genreB):
    genre_collection = get_genre_collection()
    numA = genre_collection.find_one({'Name': genreA})['PiecesNum']
    numB = genre_collection.find_one({'Name': genreB})['PiecesNum']
    print(numA, numB)
    # limit_num = get_smaller


class MixedSourceDataset(data.Dataset):
    def __init__(self):
        sources = ['metal', 'punk', 'folk', 'newage', 'country', 'bluegrass']
        data = generate_sparse_matrix_from_multiple_genres((sources))
        self.data = np.expand_dims(data, axis=1)
        self.length = data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.length


class SteelyDataset(data.Dataset):
    def __init__(self, genreA, genreB, phase, use_mix):
        assert phase in ['train', 'test'], 'not valid dataset type'

        sources = ['metal', 'punk', 'folk', 'newage', 'country', 'bluegrass']

        genre_collection = get_genre_collection()

        self.data_path = 'D:/data/'

        train_num = min(genre_collection.find_one({'Name': genreA})['TrainPieces'],
                        genre_collection.find_one({'Name': genreB})['TrainPieces'])
        test_num = min(genre_collection.find_one({'Name': genreA})['TestPieces'],
                       genre_collection.find_one({'Name': genreB})['TestPieces'])
        if phase is 'train':
            self.length = train_num

            if use_mix:
                dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA, phase)[:self.length], 1)
                dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB, phase)[:self.length], 1)
                mixed = generate_sparse_matrix_from_multiple_genres(sources)
                np.random.shuffle(mixed)
                data_mixed = np.expand_dims(mixed[:self.length], 1)

                self.data = np.concatenate((dataA, dataB, data_mixed), axis=1)

            else:
                dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA, phase)[:self.length], 1)
                dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB, phase)[:self.length], 1)
                print(dataA.shape)

                self.data = np.concatenate((dataA, dataB), axis=1)
        else:
            self.length = test_num
            dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA, phase)[:self.length], 1)
            dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB, phase)[:self.length], 1)

            self.data = np.concatenate((dataA, dataB), axis=1)

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.length

    def get_data(self):
        return self.data


class ClassifierDataset(data.Dataset):
    def __init__(self, genreA, genreB, phase):

        genre_collection = get_genre_collection()

        train_num = min(genre_collection.find_one({'Name': genreA})['TrainPieces'],
                        genre_collection.find_one({'Name': genreB})['TrainPieces'])
        test_num = min(genre_collection.find_one({'Name': genreA})['TestPieces'],
                       genre_collection.find_one({'Name': genreB})['TestPieces'])

        if phase is 'train':
            # self.length = min(20000, train_num)
            self.length = train_num
            print(self.length)

            dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA, phase)[:self.length], 1)
            dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB, phase)[:self.length], 1)

            labelA = np.array([[1.0, 0.0] for _ in range(self.length)])
            labelB = np.array([[0.0, 1.0] for _ in range(self.length)])

            self.data = np.concatenate((dataA, dataB), axis=0)
            self.labels = np.concatenate((labelA, labelB), axis=0)
            self.data_pair = [pair for pair in zip(self.data, self.labels)]

        else:
            self.length = test_num

            dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA, phase)[:self.length], 1)
            dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB, phase)[:self.length], 1)

            labelA = np.array([[1.0, 0.0] for _ in range(self.length)])
            labelB = np.array([[0.0, 1.0] for _ in range(self.length)])

            self.data = np.concatenate((dataA, dataB), axis=0)
            self.labels = np.concatenate((labelA, labelB), axis=0)
            self.data_pair = [pair for pair in zip(self.data, self.labels)]

    def __getitem__(self, index):
        return self.data_pair[index]

    def __len__(self):
        return self.length * 2

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


if __name__ == '__main__':
    dataset = SteelyDataset('rock', 'jazz', 'train', False)