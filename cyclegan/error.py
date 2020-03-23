
class CyganException(Exception):
    def __init__(self, info):
        super().__init__()
        self.info = info
    def __str__(self):
        return self.info