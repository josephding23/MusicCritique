import logging
import colorlog


class TerminalLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.setLevel(logging.INFO)
        ch = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(fg_cyan)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        ch.setFormatter(color_formatter)
        self.addHandler(ch)

    def add_file_logger(self, path):
        fh = logging.FileHandler(filename=path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.addHandler(fh)
