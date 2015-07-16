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
    
