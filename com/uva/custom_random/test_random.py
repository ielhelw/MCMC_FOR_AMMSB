import sys
from sets import Set

# import CustomRandom
import custom_random

class Edge(object):
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def __lt__(self, other):
        return self.first < other.first or (self.first == other.first and self.second < other.second)
    def __eq__(self, other):
        return self.first == other.first and self.second == other.second
    def __hash__(self):
        return self.first + self.second


rgen = custom_random.CustomRandom(42)

sys.stdout.write("gamma(1.0, 1.0):\n")
a = rgen.gamma(1.0, 1.0, 2, 4)
for r in a:
    for c in r:
        sys.stdout.write("%.12f " % c)
    sys.stdout.write("\n")

sys.stdout.write("randn:\n")
a = rgen.randn(2, 4)
for r in a:
    for c in r:
        sys.stdout.write("%.12f " % c)
    sys.stdout.write("\n")

sys.stdout.write("randint(2**20):\n")
for i in xrange(64):
    sys.stdout.write("%d " % rgen.randint(0, 2**20))
    if (i + 1) % 8 == 0:
        sys.stdout.write("\n");
sys.stdout.write("\n");

sys.stdout.write("Create random graph:\n")
graph = Set()
for i in xrange(64):
    a = rgen.randint(0, 1024)
    b = rgen.randint(0, 1024)
    graph.add(Edge(a, b))
for e in sorted(graph):
    sys.stdout.write("(%d,%d) " % (e.first, e.second))
sys.stdout.write("\n");

sys.stdout.write("Sample subgraph:\n");
subgraph = rgen.sample(graph, 32)
for e in sorted(subgraph):
    sys.stdout.write("(%d,%d) " % (e.first, e.second))
sys.stdout.write("\n");
