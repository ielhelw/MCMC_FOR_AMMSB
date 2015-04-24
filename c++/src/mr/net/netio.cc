#include <mr/net/netio.h>

// #include <mr/log.h>


namespace mr {

namespace net {

static const bool FAKE = false;

void Network::setupConnections(RWList &rwList,
							   const PeerVector &peerVector,
							   network_type::Type type,
							   bool isStreamer,
							   ::size_t me, ::size_t size,
							   ::size_t start, ::size_t size_2pow) {
	if (me < start + size_2pow) {
		for (::size_t p = 0; p < size_2pow; p++) {
			::size_t peer = start + size_2pow + (me + p) % size_2pow;
			if (FAKE) {
				std::cerr << me << ": subsize " << size_2pow << " start " << start << " " << ((peer >= size) ? "dont" : "") << " connect/forward to " << peer << std::endl;
			} else if (peer < size) {
				ReaderInterface *r = createReader(*peerVector[peer], type, isStreamer);
				WriterInterface *w = createWriter(*peerVector[peer], type, isStreamer);
				rwList[peer].reader = r;
				rwList[peer].writer = w;
				std::cerr << "Connect/fwd to peer " << peer << " reader " << r << " writer " << w << std::endl;
			}
		}
	} else {
		for (::size_t p = 0; p < size_2pow; p++) {
			::size_t peer = start + (me - p) % size_2pow;
			if (FAKE) {
				std::cerr << me << ": subsize " << size_2pow << " start " << start << " " << std::string((peer >= size) ? "dont" : "") << " connect/backward to " << peer << std::endl;
			} else if (peer < size) {
				WriterInterface *w = createWriter(*peerVector[peer], type, isStreamer);
				ReaderInterface *r = createReader(*peerVector[peer], type, isStreamer);
				rwList[peer].reader = r;
				rwList[peer].writer = w;
				std::cerr << "Connect/bck to peer " << peer << " reader " << r << " writer " << w << std::endl;
			}
		}
	}
}


void Network::createConnectionDC(RWList &rwList,
								 const PeerVector &peerVector,
								 network_type::Type type,
								 bool isStreamer,
								 ::size_t me, ::size_t size,
								 ::size_t start, ::size_t size_2pow) {
	if (size_2pow < 2) {
		return;
	}

	size_2pow /= 2;

	setupConnections(rwList, peerVector, type, isStreamer, me, size, start, size_2pow);

	// Divide and conquer
	if (me < start + size_2pow) {
		createConnectionDC(rwList, peerVector, type, isStreamer, me, size, start, size_2pow);
	} else {
		createConnectionDC(rwList, peerVector, type, isStreamer, me, size, start + size_2pow, size_2pow);
	}
}


void Network::createAllToAllConnections(RWList &rwList,
										const PeerVector &peerSet,
										network_type::Type type,
										bool isStreamer) {
	PeerVector peerVector;

	::size_t rank = std::numeric_limits< ::size_t>::max();
	::size_t size = peerSet.size();
	::size_t i = 0;
	for (PeerVector::const_iterator iter = peerSet.begin();
		 	iter != peerSet.end();
			iter++, i++) {
		peerVector.push_back(*iter);
		if (getMe() == *iter) {
			rank = i;
		}
	}
	if (rank == std::numeric_limits< ::size_t>::max()) {
		throw NetworkException("Could not find myself in the PeerVector");
	}

	::size_t size_2pow = next_2_power(size);

	rwList.clear();
	rwList.resize(peerSet.size());
	createConnectionDC(rwList, peerVector, type, isStreamer, rank, size, 0, size_2pow);
}

}

}
