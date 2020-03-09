import os

class Config(object):
    def __init__(self):
        self.name = 'steely_gan'
        self.dataset_name = 'free_midi_library'

        self.genreA = 'rock'
        self.genreB = 'jazz'
        self.save_path = 'd:/checkpoints/' + self.name
        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'
        self.test_path = self.save_path + '/test_results'

        self.G_A2B_save_path = self.model_path + '/G_A2B/'
        self.G_B2A_save_path = self.model_path + '/G_B2A/'
        self.D_A_save_path = self.model_path + '/D_A/'
        self.D_B_save_path = self.model_path + '/D_B/'

        self.image_pool_max_size = 50

        self.max_dataset_size = 10000
        self.dataset_mode = 'unaligned'

        self.track_merged = False
        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.time_step = 120
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84
        self.instr_num = 5

        self.data_shape = (16, 1, 64, 84)
        self.input_shape = (1, 64, 84)

        self.gpu = True
        '''
        self.preprocess = 'resize_and_crop'
        self.load_size = 320
        self.crop_size = 220
        self.no_flip = True
        self.num_threads = 4
        '''
        self.beta1 = 0.5                     # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.lr = 0.0002

        self.g_lr = 2e-4                     # generator learning rate
        self.d_lr = 2e-4                     # discriminator learning rate

        self.phase = 'train'
        self.no_flip = True
        self.num_threads = 4
        self.batch_size = 16
        self.max_epoch = 30
        self.epoch_step = 10

        self.plot_every = 100  # iterations
        self.save_every = 5  # epochs

        self.model = 'full' # three different models, base, partial, full

        self.continue_train = False
        self.start_epoch = 0

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)