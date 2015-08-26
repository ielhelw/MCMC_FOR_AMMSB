#include "mcmc/data.h"
#include "mcmc/fileio.h"

#include <fstream>

namespace mcmc {

// FIXME: does not belong here, but in ? np? misc? stats?
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

Edge::Edge(std::istream &s) { (void)get(s); }

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

std::ostream& dump(std::ostream& out, const NetworkGraph& graph) {
  for (auto e : graph) {
    if (e.first < e.second) {
      out << e.first << "\t" << e.second << std::endl;
    }
  }

  return out;
}

std::ostream& dump(std::ostream& out, const EdgeMap &s) {
  for (auto e = s.begin(); e != s.end(); e++) {
    out << e->first << ": " << e->second << std::endl;
  }

  return out;
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
  dump(std::cout, *E);
}

void Data::save(const std::string &filename, bool compressed) const {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  FileHandle f(filename, compressed, "w");
  int32_t num_nodes = N;
  f.write_fully(&num_nodes, sizeof num_nodes);
  for (auto r : E->edges_at_) {
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
