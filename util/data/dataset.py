import torch.utils.data as data
from util.data.create_database import generate_data_from_sparse_data, get_genre_pieces


def get_dataset(opt):
    data_path = opt.data_path
    genreA = opt.genreA
    genreB = opt.genreB
    # limit_num = get_smaller



class SteelyDataset(data.Dataset):
    def __init__(self, data_path, genre, limit_num):
        self.data_path = data_path
        self.genre = genre
        self.parts_num = 10
        self.limit_num = limit_num

        self.pieces_num = get_genre_pieces(self.genre)

        self.data = generate_data_from_sparse_data(self.data_path, self.genre, self.parts_num)

    def __getitem__(self, index):
        return self.data[index, :, :, :, :]

    def __len__(self):
        return self.pieces_num

if __name__ == '__main__':
    pass
    # dataset = SteelyDataset('d:/data', 'rock', parts_num=1)
    # print(dataset[4].shape)
    # print(len(dataset))