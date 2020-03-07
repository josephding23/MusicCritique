import os

class Config(object):
    def __init__(self):
        self.name = 'steely_gan'
        self.dataset_name = 'free_midi_library'

        self.data_root = 'd:/data'
        self.genreA = 'rock'
        self.genreB = 'jazz'
        self.save_path = './checkpoints/' + self.name
        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'
        self.test_path = self.save_path + '/test_results'

        self.max_dataset_size = 10000
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'

        self.track_merged = False
        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.n_tracks = 10000
        self.beat_resolution = 1

        self.time_step = 120
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84
        self.instr_num = 5

        if not self.track_merged:
            self.data_shape = (self.bar_length, self.time_step, self.note_valid_length, self.instr_num)
        else:
            self.data_shape = (self.bar_length, self.time_step, self.note_valid_length)

        self.gpu = True
        '''
        self.preprocess = 'resize_and_crop'
        self.load_size = 320
        self.crop_size = 220
        self.no_flip = True
        self.num_threads = 4
        '''
        self.beta1 = 0.5                     # Adam optimizer beta1

        self.g_lr = 2e-4                     # generator learning rate
        self.d_lr = 2e-4                     # discriminator learning rate

        self.phase = 'train'
        self.no_flip = True
        self.num_threads = 4
        self.batch_size = 1
        self.max_epochs = 100

        self.plot_every = 100  # iterations
        self.save_every = 10  # epochs

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)