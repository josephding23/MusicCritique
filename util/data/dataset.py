import torch.utils.data as data
from pymongo import MongoClient
import numpy as np
from torch.utils.data import DataLoader
import torch

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

        numA = genre_collection.find_one({'Name': genreA})['ValidPiecesNum']
        numB = genre_collection.find_one({'Name': genreB})['ValidPiecesNum']

        train_num = int(min(numA, numB) * 0.9)
        test_num = min(numA, numB) - train_num
        if phase is 'train':
            self.length = train_num

            if use_mix:
                dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA)[:self.length], 1)
                dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB)[:self.length], 1)
                mixed = generate_sparse_matrix_from_multiple_genres(sources)
                np.random.shuffle(mixed)
                data_mixed = np.expand_dims(mixed[:self.length], 1)

                self.data = np.concatenate((dataA, dataB, data_mixed), axis=1)

            else:
                dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA)[:self.length], 1)
                dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB)[:self.length], 1)

                self.data = np.concatenate((dataA, dataB), axis=1)
        else:
            self.length = test_num
            dataA = np.expand_dims(generate_sparse_matrix_of_genre(genreA)[:self.length], 1)
            dataB = np.expand_dims(generate_sparse_matrix_of_genre(genreB)[:self.length], 1)

            self.data = np.concatenate((dataA, dataB), axis=1)


    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = SteelyDataset('rock', 'jazz', 'train', True)
    print(dataset.data.shape)