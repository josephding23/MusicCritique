import torch.utils.data as data
from util.data.create_database import generate_data_from_sparse_data, get_genre_pieces

class SteelyDataset(data.Dataset):
    def __init__(self, opt):
        self.data_path = opt.data_path
        self.genreA = opt.genreA
        self.genreB = opt.genreB
        self.parts_num = 10

        self.pieces_num = get_genre_pieces(self.genreA)

        self.data = generate_data_from_sparse_data(self.data_path, self.genreA, self.parts_num)

    def __getitem__(self, index):
        return self.data[index, :, :, :, :]

    def __len__(self):
        return self.pieces_num

if __name__ == '__main__':
    pass
    # dataset = SteelyDataset('d:/data', 'rock', parts_num=1)
    # print(dataset[4].shape)
    # print(len(dataset))