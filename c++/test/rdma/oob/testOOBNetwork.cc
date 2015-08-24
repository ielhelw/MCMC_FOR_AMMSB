#include <dkvstore/OOBNetwork.h>

int main(int argc, char *argv[]) {
  DKV::DKVRDMA::OOBNetwork<int32_t> oob_network;

  ::size_t my_rank;
  ::size_t num_hosts = 0;       // auto-initialize please
  oob_network.Init("", 0, &num_hosts, &my_rank);

  std::vector<int32_t> info(num_hosts);
  std::vector<int32_t> peer_info(num_hosts);
  oob_network.exchange_oob_data(info, &peer_info);

  oob_network.close();

  return 0;
}

