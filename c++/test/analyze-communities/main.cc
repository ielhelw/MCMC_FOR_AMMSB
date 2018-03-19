#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <mcmc/config.h>
#include <mcmc/types.h>
#include <mcmc/data.h>

// typedef std::set<mcmc::Vertex> community_t;     // ordered set
typedef std::unordered_set<mcmc::Vertex> community_t;

struct Stats {
  float         mean;
  float         stdev;
};

static double stdev(double sum, double sumsq, ::size_t n) {
  return std::sqrt((sumsq - sum * sum / n) / (n - 1));
}

class Pi {
 public:
  Pi(const std::string &filename, ::size_t K = 0) : K_(K), filename_(filename) {
    file_has_K_ = K != 0;
  }

  void get_info() {
    std::ifstream saved;
    saved.open(filename_, std::ios::in | std::ios::binary);
    read_header(saved);
  }

  void load() {
    std::ifstream saved;
    saved.open(filename_, std::ios::in | std::ios::binary);
    read_header(saved);

    pi_.resize(N_);
    for (auto &pi: pi_) {
      pi.resize(K_ + 1);
    }
    ::size_t stored;
    for (std::size_t i = 0; i < N_; ++i) {
      saved.read(reinterpret_cast<char *>(&stored), sizeof stored);
      saved.read(reinterpret_cast<char *>(pi_[stored].data()),
                 (K_ + 1) * sizeof pi_[stored][0]);
    }
  }

  void save(const std::string &ofile) {
    std::ofstream save;
    boost::filesystem::path dir(ofile);
    if (dir.parent_path() != "") {
      boost::filesystem::create_directories(dir.parent_path());
    }
    save.open(ofile, std::ios::out | std::ios::binary);
    std::cerr << "Save pi to file " << ofile << std::endl;
    std::cerr << "mpi rank " << mpi_rank_ << " size " << mpi_size_ << std::endl;
    save.write(reinterpret_cast<char *>(&N_), sizeof N_);
    save.write(reinterpret_cast<char *>(&K_), sizeof K_);
    save.write(reinterpret_cast<char *>(&hosts_pi_), sizeof hosts_pi_);
    save.write(reinterpret_cast<char *>(&mpi_size_), sizeof mpi_size_);
    save.write(reinterpret_cast<char *>(&mpi_rank_), sizeof mpi_rank_);
    // padding
    for (auto i = 0; i < 3; i++) {
      save.write(reinterpret_cast<char *>(&i), sizeof i);
    }
    for (::size_t i = 0; i < N_; ++i) {
      save.write(reinterpret_cast<char *>(&i), sizeof i);
      save.write(reinterpret_cast<char *>(pi_[i].data()),
                 (K_ + 1) * sizeof pi_[i][0]);
    }
    save.close();
    std::cerr << "Saved pi to file " << ofile << std::endl;
  }

  float bin_flattened(std::size_t steps, std::size_t total_memberships) {
    bin_.clear();
    bin_.resize(steps, 0);
    mcmc::Float min = 1.0;
    mcmc::Float max = 0.0;
    for (auto pi: pi_) {
      for (::size_t k = 0; k < K_; ++k) {
        if (pi[k] < 0.0 || pi[k] > 1.0) {
          std::cerr << "Ouch, pi[k] out of range: " << pi[k] << std::endl;
        }
        min = std::min(min, pi[k]);
        max = std::max(max, pi[k]);
        ::size_t b = pi[k] * steps;
        // std::cerr << "pi[" << k << "] " << pi[k] << " bin " << b << std::endl;
        ++bin_[b];
      }
    }
    std::cout << "pi range: " << min << " .. " << max << std::endl;
    std::vector<std::size_t> cumul(steps, 0);
    cumul[steps - 1] = bin_[steps - 1];
    for (::size_t i = steps - 1; i > 0; --i) {
      cumul[i - 1] = cumul[i] + bin_[i - 1];
    }
    float cutoff = 1.0;
    for (::size_t i = 0; i < steps; ++i) {
      if (cumul[i] < total_memberships) {
        std::cout << "At bin " << i <<
          " pi " << static_cast<mcmc::Float>(i) / steps <<
          " cumul " << cumul[i] <<
          " #comm * <comm size> " << total_memberships << std::endl;
        float x_1 = mcmc::Float(i) / steps;
        float y_1 = mcmc::Float(cumul[i]);
        float x_0;
        float y_0;
        if (i > 0) {
          x_0 = mcmc::Float(i - 1) / steps;
          y_0 = mcmc::Float(cumul[i - 1]);
          std::cout << "At bin " << i - 1 <<
            " pi " << static_cast<mcmc::Float>((i - 1)) / steps <<
            " cumul " << cumul[i - 1] <<
            " #comm * <comm size> " << total_memberships << std::endl;
        } else {
          x_0 = 0.0;
          y_0 = 1.0;
        }
        cutoff = x_0 + (total_memberships - y_0) * (x_1 - x_0) / (y_1 - y_0);
        // std::cerr << "cutoff " << cutoff << std::endl;
        break;
      }
    }

    if (false) {
      for (auto b: bin_) {
        std::cout << b << std::endl;
      }
    }

    return cutoff;
  }

  const std::vector<std::size_t> &bin() const {
    return bin_;
  }

  void save_bin(const std::string &bin_file) const {
    std::ofstream cf;
    boost::filesystem::path dir(bin_file);
    if (dir.parent_path() != "") {
      boost::filesystem::create_directories(dir.parent_path());
    }
    cf.open(bin_file, std::ios::out);
    std::cerr << "Save bin to file " << bin_file << std::endl;
    for (auto b: bin_) {
      cf << b << std::endl;
    }
    cf.close();
  }

  Stats stats(void) const {
    Stats stats;
    double sum = 0;
    double sumsq = 0;
    if (incremental_read_) {
      std::ifstream saved;
      saved.open(filename_, std::ios::in | std::ios::binary);
      skip_header(saved);
      ::size_t buf_size = mem_avail_ / (K_ * sizeof(float));
      std::cerr << "Buffer size " << buf_size << " lines" << std::endl;
      std::vector<std::vector<float>> buffer(buf_size, std::vector<float>(K_));

      ::size_t processed = 0;
      while (processed < N_) {
        ::size_t n = std::min(N_ - processed, buf_size);
        std::cerr << "Now read buffer size " << n << " starting from " << processed << std::endl;
        for (std::size_t i = 0; i < n; ++i) {
          ::size_t stored;
          saved.read(reinterpret_cast<char *>(&stored), sizeof stored);
          saved.read(reinterpret_cast<char *>(buffer[i].data()),
                     (K_ + 1) * sizeof buffer[i][0]);
        }
        for (::size_t i = 0; i < n; ++i) {
          for (::size_t k = 0; k < K_; ++k) {
            sum += buffer[i][k];
            sumsq += buffer[i][k] * buffer[i][k];
          }
        }
        processed += n;
      }

    } else {
      for (auto pi: pi_) {
        for (::size_t k = 0; k < K_; ++k) {
          sum += pi[k];
          sumsq += pi[k] * pi[k];
        }
      }
      assert(pi_.size() == N_);
    }

    ::size_t n = N_ * K_;
    stats.stdev = (float)stdev(sum, sumsq, n);
    stats.mean  = (float)(sum / n);
    std::cerr << "N " << n << " sum " << sum << " sumsq " << sumsq << std::endl;

    return stats;
  }

  std::vector<mcmc::Vertex> read_nodemap(const std::string &file) {
    std::ifstream nmp;
    nmp.open(file, std::ios::in);
    std::vector<mcmc::Vertex> nodemap(N_, -1);

    while (! nmp.eof()) {
      mcmc::Vertex x;
      mcmc::Vertex y;
      nmp >> x >> y;
      nodemap[x] = y;
    }

    return nodemap;
  }

  std::vector<community_t> assign_communities(float cutoff) {
    std::vector<community_t> comms(K_);

    for (::size_t i = 0; i < N_; ++i) {
      for (::size_t k = 0; k < K_; ++k) {
        if (pi_[i][k] >= cutoff) {
          comms[k].insert(i);
        }
      }
    }

    return comms;
  }

  void save_communities(const std::string &communities_file, const std::vector<community_t> &communities, const std::vector<mcmc::Vertex> &nodemap) const {
    std::ofstream cf;
    boost::filesystem::path dir(communities_file);
    if (dir.parent_path() != "") {
      boost::filesystem::create_directories(dir.parent_path());
    }
    cf.open(communities_file, std::ios::out);
    std::cerr << "Save communities to file " << communities_file << std::endl;
    if (nodemap.size() == 0) {
      std::cerr << "******** FIXME need to apply node mapping" << std::endl;
      for (auto &c: communities) {
        for (auto n: c) {
          cf << n << " ";
        }
        cf << std::endl;
      }
    } else {
      for (auto &c: communities) {
        std::set<mcmc::Vertex> comm;
        for (auto n: c) {
          comm.insert(nodemap[n]);
        }
        for (auto n: comm) {
          cf << n << " ";
        }
        cf << std::endl;
      }
    }
    cf.close();
  }

  ::size_t N(void) const {
    return N_;
  }

  ::size_t K(void) const {
    return K_;
  }

  void set_incremental_read(bool incremental_read, ::size_t mem_avail) {
    incremental_read_ = incremental_read;
    mem_avail_ = mem_avail;
  }

 protected:
  ::size_t K_;
  ::size_t N_;
  std::string filename_;
  bool file_has_K_ = true;
  int32_t hosts_pi_;
  int32_t mpi_size_;
  int32_t mpi_rank_;
  std::vector<std::vector<mcmc::Float>> pi_;
  std::vector<std::size_t> bin_;
  bool incremental_read_ = false;
  ::size_t mem_avail_ = 0;

  void read_header(std::ifstream& saved) {
    saved.read(reinterpret_cast<char *>(&N_), sizeof N_);
    if (! file_has_K_) {
      saved.read(reinterpret_cast<char *>(&K_), sizeof K_);
    }
    saved.read(reinterpret_cast<char *>(&hosts_pi_), sizeof hosts_pi_);
    saved.read(reinterpret_cast<char *>(&mpi_size_), sizeof mpi_size_);
    saved.read(reinterpret_cast<char *>(&mpi_rank_), sizeof mpi_rank_);
    int32_t padding[3];
    for (auto i = 0; i < 3; ++i) {
      saved.read(reinterpret_cast<char *>(&padding[i]), sizeof padding[i]);
    }
    std::cerr << "N " << N_ << " K " << K_ << " hosts_pi " << hosts_pi_ <<
      " mpi_size " << mpi_size_ << " mpi_rank " << mpi_rank_ << std::endl;
  }

  void skip_header(std::ifstream& saved) const {
    ::size_t dummy;
    saved.read(reinterpret_cast<char *>(&dummy), sizeof dummy);
    if (! file_has_K_) {
      saved.read(reinterpret_cast<char *>(&dummy), sizeof dummy);
    }
    int32_t m;
    saved.read(reinterpret_cast<char *>(&m), sizeof m);
    saved.read(reinterpret_cast<char *>(&m), sizeof m);
    saved.read(reinterpret_cast<char *>(&m), sizeof m);
    int32_t padding[3];
    for (auto i = 0; i < 3; ++i) {
      saved.read(reinterpret_cast<char *>(&padding[i]), sizeof padding[i]);
    }
    std::cerr << "N " << N_ << " K " << K_ << " hosts_pi " << hosts_pi_ <<
      " mpi_size " << mpi_size_ << " mpi_rank " << mpi_rank_ << std::endl;
  }
};


static std::vector<community_t> read_true_communities(const std::string &filename) {
  auto compressed = boost::algorithm::ends_with(filename, ".gz");
  std::ios_base::openmode mode = std::ios_base::in;
  if (compressed) {
    mode |= std::ios_base::binary;
  }
  std::ifstream nmp(filename, mode);
  if (! nmp) {
    throw mcmc::IOException("Cannot open " + filename);
  }
  boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;

  if (compressed) {
    inbuf.push(boost::iostreams::gzip_decompressor());
  }
  inbuf.push(nmp);
  std::istream instream(&inbuf);

  std::vector<community_t> comms;
  while (! instream.eof()) {
    std::string line;
    std::getline(instream, line);

    std::stringstream ss(line);
    community_t cm;
    mcmc::Vertex vertex;
    while (ss >> vertex) {
      cm.insert(vertex);
    }
    comms.push_back(cm);
  }

  nmp.close();

  return comms;
}


static void basic_stats(std::ostream &os,
            const std::vector<community_t> &communities,
            const std::string label) {
  float sum = 0.0;
  float sqsum = 0.0;
  for (auto &c: communities) {
    auto s = static_cast<float>(c.size());
    sum += s;
    sqsum += s * s;
    // std::cerr << s << std::endl;
  }
  os << label << ": communities " << communities.size() <<
    " <size> " << (sum / communities.size()) <<
    " +- " << stdev(sum, sqsum, communities.size()) <<
    std::endl;
}


int main(int argc, char *argv[]) {
  std::size_t K = 0;
  std::string filename = "";
  std::string outfile = "";
  std::string nodemap_file = "";
  std::string communities_file = "";
  std::string true_communities_file = "";
  std::size_t bin = 0;
  std::string bin_file = "";
  std::size_t total_memberships = 0;
  std::size_t mem_avail = 32ULL << 30;
  float cutoff = -1.0;

  namespace po = boost::program_options;

  po::options_description options;
  options.add_options()
    ("input-file,f",
     po::value<std::string>(&filename),
     "input file")
    ("save,O",
     po::value<std::string>(&outfile),
     "output file")
    ("nodemap,n",
     po::value<std::string>(&nodemap_file),
     "nodemap file")
    ("communities,c",
     po::value<std::string>(&communities_file),
     "communities output file")
    ("true-communities,C",
     po::value<std::string>(&true_communities_file),
     "true communities input file")
    ("K,K",
     po::value<std::size_t>(&K),
     "force K")
    ("cutoff,x",
     po::value<float>(&cutoff),
     "force cutoff for community assingment")
    ("bin,b",
     po::value<std::size_t>(&bin),
     "bin")
    ("bin-file,B",
     po::value<std::string>(&bin_file),
     "bin dump file")
    ("total-memberships,m",
     po::value<std::size_t>(&total_memberships),
     "total number of memberships (K * average membership)")
    ("stats,S", "print statistics")
    ("mem-avail,M",
     po::value<std::size_t>(&mem_avail),
     "set available core memory")
    ("help", "print options")
    ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl;
    return 33;
  }

  if (vm.count("help") > 0) {
    std::cout << options << std::endl;
    return 0;
  }

  std::cerr << "K " << K << " input " << filename << " save " << outfile << std::endl;

  if (true_communities_file != "") {
    auto true_communities = read_true_communities(true_communities_file);
    basic_stats(std::cerr, true_communities, "true communities");
  }

  if (filename != "") {
    Pi pi(filename, K);
    pi.get_info();
    if (pi.K() * pi.N() * sizeof(float) > mem_avail) {
      pi.set_incremental_read(true, mem_avail);
    } else {
      pi.load();
      std::cerr << "Done loading " << filename << std::endl;
    }

    if (vm.count("stats") > 0) {
      auto stats = pi.stats();
      std::cerr << "Pi average " << stats.mean << " std " << stats.stdev << std::endl;
    }

    if (bin > 0) {
      float c = pi.bin_flattened(bin, total_memberships);
      if (cutoff == -1.0) {
        cutoff = c;
      }
      if (bin_file != "") {
        pi.save_bin(bin_file);
      }
    }
    std::cerr << "cutoff " << cutoff << std::endl;
    if (cutoff >= 0.0) {
      auto communities = pi.assign_communities(cutoff);
      basic_stats(std::cerr, communities, "communities from pi");
      std::vector<mcmc::Vertex> nodemap;
      if (nodemap_file != "") {
        nodemap = pi.read_nodemap(nodemap_file);
      }

      if (communities_file != "") {
        pi.save_communities(communities_file, communities, nodemap);
      }
    }

    if (outfile != "") {
      pi.save(outfile);
    }
  }

  return 0;
}
