

class BaseDataCollator:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def random_select(self):
        pass