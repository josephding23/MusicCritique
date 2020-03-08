import torch.utils.data as data
from pymongo import MongoClient
import numpy as np

from util.data.create_database import generate_sparse_matrix_from_nonzeros, get_genre_collection


def get_dataset(genreA, genreB):
    genre_collection = get_genre_collection()
    data_path = 'D:/data'
    numA = genre_collection.find_one({'Name': genreA})['PiecesNum']
    numB = genre_collection.find_one({'Name': genreB})['PiecesNum']
    print(numA, numB)
    # limit_num = get_smaller



class SteelyDataset(data.Dataset):
    def __init__(self, genreA, genreB):
        genre_collection = get_genre_collection()

        self.data_path = 'D:/data/'

        numA = genre_collection.find_one({'Name': genreA})['PiecesNum']
        numB = genre_collection.find_one({'Name': genreB})['PiecesNum']
        self.length = min(numA, numB)


        dataA = np.expand_dims(generate_sparse_matrix_from_nonzeros(genreA)[:self.length], 1)
        dataB = np.expand_dims(generate_sparse_matrix_from_nonzeros(genreB)[:self.length], 1)

        self.data = np.concatenate((dataA, dataB), axis=1)
        print(self.data.shape)


    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = SteelyDataset('rock', 'jazz')
    print(dataset[0].shape)