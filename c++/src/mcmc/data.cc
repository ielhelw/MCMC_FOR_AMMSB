#include "mcmc/data.h"
#include "mcmc/fileio.h"

#include <fstream>

namespace std {
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
int32_t hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
  int32_t h = std::hash<int32_t>()(x.first) ^ std::hash<int32_t>()(x.second);
  return h;
}
#else
::size_t hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
  ::size_t h = ((size_t)x.first * (size_t)x.second) ^
               ((size_t)x.first + (size_t)x.second);
  return h;
}
#endif
}

namespace mcmc {

void print_mem_usage(std::ostream &s) {
  static const int64_t MEGA = 1 << 20;
  static int64_t pagesize = 0;
  static std::string proc_statm;
  if (pagesize == 0) {
    pagesize = sysconf(_SC_PAGESIZE);
    std::ostringstream ss;
    ss << "/proc/" << getpid() << "/statm";
    proc_statm = ss.str();
    s << "For memory query file " << proc_statm << std::endl;
  }

  std::ifstream statm(proc_statm);
  if (!statm) {
    std::cerr << "Cannot open input file \"" << proc_statm << "\"" << std::endl;
    return;
  }

  ::size_t total;
  ::size_t resident;
  ::size_t shared;
  ::size_t text;
  ::size_t data;
  ::size_t library;
  ::size_t dirty;
  statm >> total >> resident >> shared >> text >> data >> library >> dirty;

  s << "Memory usage: total " << ((total * pagesize) / MEGA) << "MB "
    << "resident " << ((resident * pagesize) / MEGA) << "MB " << std::endl;
}

Edge::Edge() {}

Edge::Edge(Vertex a, Vertex b) : first(a), second(b) {}

Edge::Edge(std::istream &s) { (void)get(s); }

void Edge::insertMe(AdjacencyList *s) const {
  Vertex max = std::max(first, second);
  if (static_cast<::size_t>(max) >= s->size()) {
    s->resize(max + 1);
  }
  (*s)[first].insert(second);
  (*s)[second].insert(first);
}

bool Edge::operator==(const Edge &a) const {
  return a.first == first && a.second == second;
}

bool Edge::operator<(const Edge &a) const {
  return first < a.first || (first == a.first && second < a.second);
}

std::ostream &Edge::put(std::ostream &s) const {
  s << std::setw(1) << "(" << first << ", " << second << ")";
  return s;
}

char Edge::consume(std::istream &s, char expect) {
  char c;
  while (true) {
    c = s.get();
    if (isspace(c)) {
      continue;
    }
    if (c != expect) {
      std::ostringstream os;
      os << "Expect " << expect << ", get '" << c << "'";
      throw MalformattedException(os.str());
    }
    return c;
  }
}

std::istream &Edge::get(std::istream &s) {
  consume(s, '(');
  s >> first;
  consume(s, ',');
  s >> second;
  consume(s, ')');

  return s;
}

std::ostream &operator<<(std::ostream &s, const Edge &e) { return e.put(s); }

std::istream &operator>>(std::istream &s, Edge &e) { return e.get(s); }

#ifdef MCMC_USE_GOOGLE_SPARSE_HASH
bool EdgeEquals::operator()(const Edge &e1, const Edge &e2) const {
  return e1 == e2;
}
#endif

std::ostream &dump_edgeset(std::ostream &out, ::size_t N,
                           const AdjacencyList &E) {
  // out << "Edge set size " << N << std::endl;
  for (::size_t n = 0; n < E.size(); n++) {
    for (auto e : E[n]) {
      if (e > static_cast<Vertex>(n)) {
        out << n << "\t" << e << std::endl;
      }
    }
  }

  return out;
}

bool present(const AdjacencyList &s, const Edge &edge) {
  for (auto e : s[edge.first]) {
    if (e == edge.second) {
      return true;
    }
  }

  return false;
}

void dump(const EdgeMap &s) {
  for (auto e = s.begin(); e != s.end(); e++) {
    std::cout << e->first << ": " << e->second << std::endl;
  }
}

Data::Data(const void *V, const NetworkGraph *E, Vertex N,
           const std::string &header)
    : V(V), E(E), N(N), header_(header) {}

Data::~Data() {
  // delete const_cast<void *>(V); FIXME: somebody must delete V; the 'owner'
  // of this dataset, I presume
  delete const_cast<NetworkGraph *>(E);
}

void Data::dump_data() const {
  // std::cout << "Edge set size " << N << std::endl;
  std::cout << header_;
  (void)dump_edgeset(std::cout, N, *E);
}

void Data::save(const std::string &filename, bool compressed) const {
#if defined EDGESET_IS_ADJACENCY_LIST && defined MCMC_USE_GOOGLE_SPARSE_HASH
  FileHandle f(filename, compressed, "w");
  int32_t num_nodes = N;
  f.write_fully(&num_nodes, sizeof num_nodes);
  for (auto r : *E) {
    GoogleHashSet &rc = const_cast<GoogleHashSet &>(r);
    rc.write_metadata(f.handle());
    rc.write_nopointer_data(f.handle());
  }
#else
  throw MCMCException(std::string(__func__) +
                      "() not implemented for this graph representation");
#endif
}

} // namespace mcmc
