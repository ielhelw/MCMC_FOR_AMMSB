from random import Random
import numpy as np
# import random

class FileRandom(Random):
    '''
    def __new__(cls, *args, **kwargs):
        return super(FileRandom, cls).__new__(cls, random())
    '''

    def __init__(self):
        Random.__init__(self)
        self.prefix = ""
        self.random_float_file = open("random.random", "w")
        self.random_int_file = open("random.randint", "w")
        self.random_sample_file = open("random.sample", "w")
        self.random_gamma_file = open("random.gamma", "w")
        self.random_choice_file = open("random.choice", "w")
        self.random_noise_file = open("random.noise", "w")


    def seed(self, x):
        print "Set random seed to " + str(x) + "\n"
        Random.seed(self, x)
        np.random.seed(x)


    def random(self):
        r = Random.random(self)
        self.random_float_file.write("%s%.17g\n" % (self.prefix, r))

        return r;


    def randint(self, a, b):
        old = self.prefix
        self.prefix = self.prefix + "% int "
        r = Random.randint(self, a, b)
        self.prefix = old
        self.random_int_file.write("%s%d\n" % (self.prefix, r))

        return r


    def sample(self, population, k):
        old = self.prefix
        self.prefix = self.prefix + "# sample "
        s = Random.sample(self, population, k)
        for n in s:
            self.random_sample_file.write("%s\n" % str(n))
        self.prefix = old

        return s


    def choice(self, seq):
        # sys.std.write("in FileRandom.choice\n")
        old = self.prefix
        self.prefix = self.prefix + "# choice "
        c = Random.choice(self, seq)
        for n in c:
            self.random_choice_file.write("%s\n" % str(n))
        self.prefix = old

        return c;


    def gamma(self, a, b, dims):
        x = np.random.gamma(a, b, dims)
        for r in x:
            for c in r:
                self.random_gamma_file.write("%.17g " % c)
            self.random_gamma_file.write("\n")

        return x


    def randn(self, a):
        x = np.randn(a)
        self.random_noise_file.write("%.17g\n" % x)
        return x


_inst = FileRandom()
random = _inst.random
randint = _inst.randint
sample = _inst.sample
choice = _inst.choice
seed = _inst.seed
file_random = _inst

if __name__ == '__main__':
    for i in range(0, 100):
        print("%d" % file_random.randint(0, 1024))
