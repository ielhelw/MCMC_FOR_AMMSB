#ifndef MCMC_DKVSTORE_RDMA_OOB_NETWORK_H__
#define MCMC_DKVSTORE_RDMA_OOB_NETWORK_H__

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <vector>
#include <string>
#include <iostream>

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#endif
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif


namespace DKV {
namespace DKVRDMA {


class NetworkException : public std::exception {
 public:
  NetworkException(const std::string &reason) throw() : reason_(reason) {
  }

  virtual ~NetworkException() throw() {
  }

  virtual const char *what() const throw() {
    return reason_.c_str();
  }

 protected:
  const std::string &reason_;
};


class OOB {
 public:
  OOB(const std::string &server = "", uint32_t port = 0, ::size_t num_hosts = 0)
      : server_(server), port_(port), num_hosts_(num_hosts) {
    if (port_ == 0) {
      port_ = 0x3eda;
    }
    if (server_ == "") {
      auto hostnames = get_prun_env();
      server_ = hostnames[0];
    }
    if (num_hosts_ == 0) {
      auto hostnames = get_prun_env();
      num_hosts_ = hostnames.size();
    }
    hostname_ = boost::asio::ip::host_name();
  }

  static std::string getenv_str(const std::string& name) {
    std::string s;
    const char *var = getenv(name.c_str());
    if (var != NULL) {
      s = std::string(var);
    }

    return s;
  }

  static std::vector<std::string> get_prun_env() {
    std::string hosts;
    hosts = getenv_str("PRUN_PE_HOSTS");
    if (hosts == "") {
      namespace io = boost::iostreams;
      hosts = getenv_str("SLURM_NODELIST");
      if (hosts != "") {
        std::string command("scontrol show hostnames " + hosts);
        FILE *scontrol = popen(command.c_str(), "r");
        if (scontrol == NULL) {
          throw NetworkException("Cannot popen(scontrol ...)");
        }
        io::stream<io::file_descriptor_source> fpstream(fileno(scontrol),
                                                        io::close_handle);
        std::string line;
        std::vector<std::string> hostnames;
        while (std::getline(fpstream, line)) {
          hostnames.push_back(line);
        }
        return hostnames;
      }
    }
    if (hosts == "") {
      hosts = getenv_str("HOSTS");
    }
    if (hosts == "") {
      throw NetworkException("Need to set prun host names");
    }
    hosts = boost::trim_copy(hosts);
    std::vector<std::string> hostnames;
    boost::split(hostnames, hosts, boost::is_any_of(" "));

    return hostnames;
  }

  bool i_am_master() const {
    return server_ == hostname_;
  }

  std::string   server_;
  uint32_t      port_;
  ::size_t      num_hosts_;
  std::string   hostname_;
};

enum OPCODE {
  QUIT,
  BARRIER,
};

template <class PeerInfo>
class OOBNetworkServer {
 public:
  OOBNetworkServer(const OOB& oob) : oob_(oob) {
  }

  OOBNetworkServer(const OOBNetworkServer &from) : oob_(from.oob_) {
  }

  // Server thread run()
  void operator() () {
    using boost::asio::ip::tcp;

    std::vector<boost::asio::ip::tcp::socket> server_socket;

    tcp::acceptor acceptor(io_service_,
                           tcp::endpoint(tcp::v4(), oob_.port_));
    int32_t ranks = 1;
    for (::size_t i = 0; i < oob_.num_hosts_; ++i) {
      boost::system::error_code error;

      server_socket.push_back(tcp::socket(io_service_));
      acceptor.accept(server_socket[i]);

      uint32_t size;
      boost::asio::read(server_socket[i],
                        boost::asio::buffer(&size, sizeof size),
                        boost::asio::transfer_all(), error);
      if (error) {
        throw boost::system::system_error(error);
      }
      char peer_hostname[size];
      boost::asio::read(server_socket[i],
                        boost::asio::buffer(peer_hostname, size),
                        boost::asio::transfer_all(), error);
      if (error) {
        throw boost::system::system_error(error);
      }

      int32_t rank;
      if (std::string(peer_hostname) == oob_.hostname_) {
        rank = 0;
      } else {
        rank = ranks;
        ranks++;
      }
      boost::asio::write(server_socket[i],
                         boost::asio::buffer(&rank, sizeof rank),
                         boost::asio::transfer_all(),
                         error);
      if (error) {
        throw boost::system::system_error(error);
      }
    }

    std::vector<std::vector<PeerInfo>> peer_oob(oob_.num_hosts_,
                                                std::vector<PeerInfo>(oob_.num_hosts_));
    for (::size_t i = 0; i < oob_.num_hosts_; ++i) {
      boost::system::error_code error;

      boost::asio::read(server_socket[i],
                        boost::asio::buffer(peer_oob[i].data(),
                                            peer_oob[i].size() * sizeof peer_oob[i][0]),
                        boost::asio::transfer_all(), error);
    }

    for (::size_t i = 0; i < oob_.num_hosts_; ++i) {
      boost::system::error_code error;

      std::vector<PeerInfo> scatter(oob_.num_hosts_);
      for (::size_t j = 0; j < oob_.num_hosts_; ++j) {
        scatter[j] = peer_oob[j][i];
      }
      boost::asio::write(server_socket[i],
                         boost::asio::buffer(scatter.data(),
                                             scatter.size() * sizeof scatter[0]),
                         boost::asio::transfer_all(), error);
      if (error) {
        throw boost::system::system_error(error);
      }
    }

    // Receive an ack or a barrier from everybody
    std::vector<bool> quit(oob_.num_hosts_, false);
    ::size_t quitted = 0;
    ::size_t synced = 0;
    while (quitted < oob_.num_hosts_) {
      for (::size_t i = 0; i < oob_.num_hosts_; ++i) {
        boost::system::error_code error;

        int32_t opcode;
        boost::asio::read(server_socket[i],
                          boost::asio::buffer(&opcode, sizeof opcode),
                          boost::asio::transfer_all(), error);
        if (error) {
          throw boost::system::system_error(error);
        }
        switch (static_cast<OPCODE>(opcode)) {
        case OPCODE::QUIT:
          if (quit[i]) {
            throw NetworkException("Host " + std::to_string(i) +
                                   " already quit");
          }
          quit[i] = true;
          quitted++;
          break;
        case OPCODE::BARRIER:
          synced++;
          if (synced == oob_.num_hosts_) {
            synced = 0;
            // Release all workers
            for (::size_t j = 0; j < oob_.num_hosts_; ++j) {
              boost::asio::write(server_socket[j],
                                 boost::asio::buffer(&opcode, sizeof opcode),
                                 boost::asio::transfer_all(), error);
            }
          }
        }
      }
    }

    for (auto & s : server_socket) {
      s.close();
    }
  }

 private:
  const OOB &oob_;
  boost::asio::io_service io_service_;
};


template <class PeerInfo>
class OOBNetwork {
 public:
  OOBNetwork() {
  }

  ~OOBNetwork() {
    if (! acked_) {
      close();
    }

    if (oob_.i_am_master()) {
      server_thread_.join();
    }
  }


  void Init(const std::string& server_host, uint32_t port,
            ::size_t *num_hosts, ::size_t *my_rank) {
    oob_ = OOB(server_host, port, *num_hosts);

    using boost::asio::ip::tcp;

    tcp::resolver resolver(io_service_);

    if (oob_.i_am_master()) {
      // start service
      server_thread_ = boost::thread(OOBNetworkServer<PeerInfo>(oob_));
    }

    boost::system::error_code error;

    tcp::resolver::query query(oob_.server_, std::to_string(oob_.port_));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
    tcp::resolver::iterator end;

    socket_ = std::unique_ptr<tcp::socket>(new tcp::socket(io_service_));
    const int max_attempts = 1000;
    for (auto i = 0; i < max_attempts; ++i) {
      auto ep = endpoint_iterator;
      for (; ep != end; ++ep) {
        socket_->close();
        socket_->connect(*ep, error);
        if (error != boost::asio::error::host_not_found) {
          break;
        }
      }
      if (ep == end ||
          error == boost::asio::error::basic_errors::connection_refused) {
        std::cerr << "connect request to " << oob_.hostname_ << ":" <<
          oob_.port_ << " timed out" << std::endl;
        usleep(100000);
      } else if (error) {
        throw boost::system::system_error(error);
      } else {
        break;
      }
    }

    uint32_t size = oob_.hostname_.size();
    boost::asio::write(*socket_,
                       boost::asio::buffer(&size, sizeof size),
                       boost::asio::transfer_all(), error);
    if (error) {
      throw boost::system::system_error(error);
    }
    boost::asio::write(*socket_,
                       boost::asio::buffer(oob_.hostname_),
                       boost::asio::transfer_all(), error);
    if (error) {
      throw boost::system::system_error(error);
    }
    boost::asio::read(*socket_, boost::asio::buffer(&rank_, sizeof rank_),
                      boost::asio::transfer_all(),
                      error);
    if (error) {
      throw boost::system::system_error(error);
    }

    std::cerr << oob_.hostname_ << ": RDMA OOB network: receive message from master, my rank is " << rank_ << std::endl;

    *my_rank = rank_;
    *num_hosts = oob_.num_hosts_;
  }

  void exchange_oob_data(const std::vector<PeerInfo>& my_oob,
                         std::vector<PeerInfo>* peer_oob) {
    boost::system::error_code error;

    boost::asio::write(*socket_,
                       boost::asio::buffer(my_oob.data(),
                                           my_oob.size() * sizeof my_oob[0]),
                       boost::asio::transfer_all(), error);
    if (error) {
      throw boost::system::system_error(error);
    }

    peer_oob->resize(oob_.num_hosts_);
    boost::asio::read(*socket_,
                      boost::asio::buffer(peer_oob->data(),
                                          peer_oob->size() * sizeof (*peer_oob)[0]),
                      boost::asio::transfer_all(), error);
    if (error) {
      throw boost::system::system_error(error);
    }
  }

  // There is only a connection to the master. No smart barriers.
  void barrier() {
    boost::system::error_code error;

    uint32_t opcode = static_cast<uint32_t>(OPCODE::BARRIER);
    boost::asio::write(*socket_, boost::asio::buffer(&opcode, sizeof opcode),
                       boost::asio::transfer_all(),
                       error);
    if (error) {
      throw boost::system::system_error(error);
    }
    boost::asio::read(*socket_, boost::asio::buffer(&opcode, sizeof opcode),
                      boost::asio::transfer_all(),
                      error);
    if (error) {
      throw boost::system::system_error(error);
    }
  }

  void close() {
    boost::system::error_code error;

    uint32_t opcode = static_cast<uint32_t>(OPCODE::QUIT);
    boost::asio::write(*socket_,
                       boost::asio::buffer(&opcode, sizeof opcode),
                       boost::asio::transfer_all(), error);
    if (error) {
      throw boost::system::system_error(error);
    }
  }

 private:
  OOB oob_;
  uint32_t rank_;
  boost::asio::io_service io_service_;
  boost::thread server_thread_;
  std::unique_ptr<boost::asio::ip::tcp::socket> socket_ = nullptr;
  bool acked_ = false;
};

}       // namespace DKVRDMA
}       // namespace DKV

#endif  // ndef MCMC_DKVSTORE_RDMA_OOB_NETWORK_H__
