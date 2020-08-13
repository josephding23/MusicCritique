import time
import torch
import re
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
import os
from util.data.dataset import ClassifierDataset
import torch.nn as nn
from torchsummary import summary
from torchnet.meter import MovingAverageValueMeter
import shutil
from classify.old_network import Classifier
from classify.new_network import NewClassifier
from classify.classify_config import ClassifierConfig
from util.logger import TerminalLogger

class Classify(object):
    def __init__(self, opt, device_name):
        self.opt = opt

        self.device_name = device_name
        if self.device_name == 'TPU':
            import torch_xla
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device() 
        else:
            self.device = torch.device('cuda')

        self.logger = TerminalLogger('logger')

        self._build_model()

    def _build_model(self):
        self.classifier = NewClassifier()

        if self.opt.gpu:
            self.classifier.to(self.device)
            # summary(self.classifier, input_size=self.opt.input_shape)

        self.classifier_optimizer = Adam(params=self.classifier.parameters(), lr=self.opt.lr,
                                         betas=(self.opt.beta1, self.opt.beta2),
                                         weight_decay=self.opt.weight_decay)

        self.classifier_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.classifier_optimizer, T_0=1, T_mult=2, eta_min=4e-08)

    def save_model(self, epoch):
        classifier_filename = f'C_{self.opt.genreA}_{self.opt.genreB}_{epoch}.pth'
        classifier_filepath = os.path.join(self.opt.checkpoint_path, classifier_filename)

        print(classifier_filepath)
        
        if epoch - self.opt.save_every >= 0:
            old_classifier_filename = f'C_{self.opt.genreA}_{self.opt.genreB}_{epoch - self.opt.save_every}.pth'

            os.remove(old_classifier_filename)

        torch.save(self.classifier.state_dict(), classifier_filename)

        self.logger.info('model saved')

    def find_latest_checkpoint(self):
        path = self.opt.checkpoint_path
        file_list = os.listdir(path)
        match_str = r'\d+'
        epoch_list = sorted([int(re.findall(match_str, file)[0]) for file in file_list])

        print(epoch_list)

        if len(epoch_list) == 0:
            raise Exception('No model to load.')
        latest_num = epoch_list[-1]
        return latest_num

    def continue_from_latest_checkpoint(self):
        latest_checked_epoch = self.find_latest_checkpoint()
        self.opt.start_epoch = latest_checked_epoch + 1

        classifier_filename = f'C_{self.opt.genreA}_{self.opt.genreB}_{latest_checked_epoch}.pth'

        self.classifier.load_state_dict(torch.load(classifier_filename))

        print(f'Loaded model from epoch {self.opt.start_epoch-1}')

    def reset_save(self):
        if os.path.exists(self.opt.save_path):
            shutil.rmtree(self.opt.save_path)

        os.makedirs(self.opt.save_path, exist_ok=True)
        os.makedirs(self.opt.model_path, exist_ok=True)
        os.makedirs(self.opt.checkpoint_path, exist_ok=True)
        os.makedirs(self.opt.test_path, exist_ok=True)

    def train(self):
        torch.cuda.empty_cache()

        ######################
        # Save / Load model
        ######################

        if self.opt.continue_train:
            try:
                self.continue_from_latest_checkpoint()
            except Exception as e:
                self.logger.error(e)
                return

        else:
            self.reset_save()

        self.logger.add_file_logger(self.opt.log_path)

        ######################
        # Dataset
        ######################

        dataset = ClassifierDataset(self.opt.genreA, self.opt.genreB, 'train')

        test_dataset = ClassifierDataset(self.opt.genreA, self.opt.genreB, 'test')

        dataset_size = len(dataset)
        iter_num = int(dataset_size / self.opt.batch_size)

        plot_every = iter_num // 10

        self.logger.info(
            f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Initiate
        ######################

        softmax_criterion = nn.BCELoss()

        Loss_meter = MovingAverageValueMeter(self.opt.plot_every)

        losses = {}

        ######################
        # Start Training
        ######################

        test_data = torch.from_numpy(test_dataset.get_data()).to(self.device, dtype=torch.float)

        gaussian_noise = torch.normal(mean=torch.zeros(test_data.shape), std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
        # test_data += gaussian_noise

        real_test_label = torch.from_numpy(test_dataset.get_labels()).view(-1, 2).to(self.device, dtype=torch.float)

        for epoch in range(self.opt.start_epoch, self.opt.max_epoch):
            loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_threads, drop_last=False)
            epoch_start_time = time.time()

            for i, batch in enumerate(loader):
                data = batch[0].to(self.device, dtype=torch.float)

                real_label = batch[1].view(self.opt.batch_size, 2).to(self.device, dtype=torch.float)

                self.classifier_optimizer.zero_grad()

                estimate_train = self.classifier(data)

                loss = softmax_criterion(estimate_train, real_label)

                loss.backward()

                self.classifier_optimizer.step()

                Loss_meter.add(loss.item())

                # test
                if i % plot_every == 0:
                    with torch.no_grad():
                        estimate_test = self.classifier(test_data)
                    estimate_test = nn.functional.softmax(estimate_test, dim=1)
                    test_prediction = torch.argmax(estimate_test, 1).eq(torch.argmax(real_test_label, 1))
                    test_accuracy = torch.mean(test_prediction.type(torch.float32)).cpu()

                    self.logger.info('Epoch {} progress {:.2%}: Loss: {}, Accuracy: {}\n'.format(epoch, i / iter_num, Loss_meter.value()[0], test_accuracy))

            if epoch % self.opt.save_every == 0 or epoch == self.opt.max_epoch - 1:
                self.save_model(epoch)

            self.classifier_scheduler.step(epoch)

            epoch_time = int(time.time() - epoch_start_time)
            self.logger.info(f'Epoch {epoch} finished, cost time {epoch_time}\n')


def train():
    opt = ClassifierConfig(1)
    classifiy = Classify(opt)
    classifiy.train()


if __name__ == '__main__':
    train()