import timeit
import sys

class Timer(object):
    def __init__(self, name):
        self._name = name
        self._N = 0
        self._total = 0

    def start(self):
        self._start = timeit.default_timer()

    def stop(self):
        self._total = timeit.default_timer() - self._start
        self._N += 1

    def print_header(self, out = sys.stdout):
        out.write("%-36s %12s %8s %14s\n" % ("timer", "total (s)", "ticks", "per tick (us)"))

    def report(self, out = sys.stdout):
        out.write("%-36s " % self._name)
        if self._N == 0:
            out.write("<unused>")
        else:
            out.write("%12.3f " % (self._total))
            out.write("%8d " % (self._N))
            out.write("%14.3f" % ((self._total / self._N) * 1000000.0))
        out.write("\n")
        
        
