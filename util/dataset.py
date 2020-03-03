import torch.utils.data as data
from util.create_database import generate_data_from_sparse_data, get_genre_pieces

class SteelyDataset(data.Dataset):
    def __init__(self, data_path, genre, parts_num=10):
        self.data_path = data_path
        self.genre = genre
        self.parts_num = parts_num

        self.pieces_num = get_genre_pieces(genre)

        self.data = generate_data_from_sparse_data(data_path, genre, parts_num)

    def __getitem__(self, index):
        return self.data[index, :, :, :, :]

    def __len__(self):
        return self.pieces_num

if __name__ == '__main__':
    dataset = SteelyDataset('d:/data', 'rock', parts_num=1)
    print(dataset[4].shape)
    print(len(dataset))