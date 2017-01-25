import time


class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print("%s takes %d s" % (self.name, self.interval))
