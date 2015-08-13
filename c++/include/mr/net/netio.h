#ifndef MR_NET_NETIO_H__
#define MR_NET_NETIO_H__

#include <inttypes.h>

#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>

#include <mr/io.h>
// #include <mr/util.h>
#include <mcmc/exception.h>

namespace mr {

/* Imported from mr/util.h */
typedef std::pair<std::string, std::string> Option;
typedef std::vector<Option> OptionList;


/* Imported from mr/util.h */
// See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
inline uint32_t next_2_power(uint32_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	return v;
}


/* Imported from mr/util.h */
// See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
inline uint64_t next_2_power(uint64_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}


namespace net {

/**
 * NetworkException
 */
class NetworkException : public mcmc::IOException {
public:
	NetworkException(const std::string &reason) : mcmc::IOException(reason) {
	}
};


class ConnectionClosedException : public NetworkException {
public:
	ConnectionClosedException(const std::string &reason)
	   		: NetworkException(reason) {
	}
};


/**
 * The message types we will exchange
 */
namespace network_type {
	enum Type {
		ANY				= 0,
		MAP         	= 1025,
		MAP_ACK,
		INTERMEDIATE,
		INTERMEDIATE_ACK,
		COUNTER,
        BROADCAST,
	};
}


/**
 * Base class for network peers
 */
class NetworkPeer {
public:
	static NetworkPeer *any;
	virtual ~NetworkPeer() {
	}

	virtual std::ostream &put(std::ostream &s) const = 0;

	virtual const std::string shortName() const = 0;
};

inline std::ostream& operator<< (std::ostream& s, const NetworkPeer &peer) {
	return peer.put(s);
}


// Need forward declarations
class Network;


class NetworkEndpoint {
public:
	const NetworkPeer &getPeer() {
		return peer;
	}

protected:
	NetworkEndpoint(Network *network,
				   	const NetworkPeer &peer,
				   	network_type::Type type,
					bool isStreamer)
			: network(network), peer(peer), type(type),
			  isStreamer(isStreamer),
			  closed(false), peerClosed(false) {
	}

	Network *network;
	const NetworkPeer &peer;
	network_type::Type	type;
	bool isStreamer;
	bool closed;
	bool peerClosed;
};


/**
 * Base class for NetworkReader-s. Network readers and writers take a peer and
 * a type (message tag) as arguments.
 */
class NetworkReader : public ReaderInterface, public NetworkEndpoint {
public:
	NetworkReader(Network *network,
				  const NetworkPeer &peer,
				  network_type::Type type,
				  bool isStreamer = false)
			: NetworkEndpoint(network, peer, type, isStreamer) {
	}

	virtual ~NetworkReader() {
	}

	// @throw an IOException or NetworkException on error
	// @return 0 if the connection is closed in a regular manner
	virtual ::size_t read(void *data, ::size_t count) = 0;
	// @throw an IOException or NetworkException on error
	// @return 0 if the connection is closed in a regular manner
	virtual ::size_t readv(const struct ::iovec *iov, int iovlen) = 0;

	// @return -1 when size cannot be determined
	virtual ::ssize_t getSize() const = 0;

	virtual bool hasMore() = 0;

	virtual void close() = 0;
};


/**
 * Base class for NetworkWriter-s. Network readers and writers take a peer and
 * a type (message tag) as arguments.
 */
class NetworkWriter : public WriterInterface, public NetworkEndpoint {
public:
	NetworkWriter(Network *network,
				  const NetworkPeer &peer,
				  network_type::Type type,
				  bool isStreamer)
			: NetworkEndpoint(network, peer, type, isStreamer) {
	}

	virtual ~NetworkWriter() {
	}

	virtual void write(const void *data, size_t count) = 0;
	virtual void writev(const struct iovec *iov, int iovlen) = 0;

	virtual void close() = 0;
};


/**
 * Base class for network implementations.
 * Also serves as a factory for NetworkReader-s and NetworkWriter-s
 */
class NetworkImpl {
public:
	NetworkImpl(const std::string &name) : name(name) {
	}

	virtual ~NetworkImpl() {
	}

	const std::string &getName() const {
		return name;
	}

	void setName(const std::string &name) {
		this->name = name;
	}

	/**
	 * Creates a new Network. Must be cleaned up by delete.
	 */
	static Network *createNetwork(const std::string &name,
								  const OptionList &options = OptionList(0)) {
		Mapper::iterator n = networkMap().find(name);
		if (n == networkMap().end()) {
			throw NetworkException("implementation " +
								   	std::string(name) +
								   	" not found");
		}

		return n->second->createNetwork(options);
	}

protected:
	typedef std::map<const std::string, NetworkImpl *> Mapper;

	std::string name;

	virtual Network *createNetwork(const OptionList &options) = 0;

	// Fix to the 'static initializer order fiasco' as per
	// http://www.parashift.com/c++-faq-lite/ctors.html#faq-10.15
	static Mapper &networkMap() {
		static Mapper *networkMap = new Mapper();
		return *networkMap;
	}

	static void addNetworkImpl(NetworkImpl *impl) {
		networkMap()[impl->getName()] = impl;
	}
};



struct RWListItem {
	RWListItem() : reader(NULL), writer(NULL) {
	}

	ReaderInterface *reader;
   	WriterInterface *writer;
};


/**
 * Base class for networks. Also invokes registered NetworkImpl-s to create
 * Networks.
 */
class Network {

public:
	typedef std::vector<ReaderInterface *> ReaderList;
	typedef std::set<const NetworkPeer *> PeerSet;
	typedef std::vector<const NetworkPeer *> PeerVector;

	virtual ~Network() {
	}

	// This call may be a rendez-vous with a createWriter call at the peer's
	virtual ReaderInterface *createReader(const NetworkPeer &conn,
										  network_type::Type type,
										  bool isStreamer = false) = 0;

	// This call may be a rendez-vous with a createReader call at the peer's
	virtual WriterInterface *createWriter(const NetworkPeer &conn,
										  network_type::Type type,
										  bool isStreamer = false) = 0;

	virtual void probe(ReaderList &readers,
					   network_type::Type type = network_type::ANY,
					   bool block = true) = 0;

	const PeerVector &getPeers() const {
		return peers;
	}

	virtual const NetworkPeer *getMe() const = 0;

	virtual const NetworkPeer *getMaster() const {
		return *peers.begin();
	}

	virtual std::ostream &put(std::ostream &s) const = 0;

	typedef std::vector<RWListItem> RWList;

	// Utility to make an all-to-all connected network in parallel, with
	// rendez-vous style connect/accepts
	// Fills rwList with a pair of { ReaderInterface, WriterInterface }
	// endpoints, connected to each of the peers in peerSet.
	// This is a collective call. Entry rwList[i] matches a connection to
	// peerSet[i], for each of the participants. A connection to self is not
	// established; the reader and writer in rwList[self] are NULL.
	void createAllToAllConnections(RWList &rwList,
								   const PeerVector &peerSet,
								   network_type::Type type,
								   bool isStreamer = false);

	void setVerbose(bool on) {
		verbose = on;
	}

	inline bool getVerbose() const {
		return verbose;
	}

protected:
	Network() : verbose(false) {
	}

	// Utility to make an all-to-all connected network in parallel, with
	// rendez-vous style connect/accepts
	void setupConnections(RWList &rwList,
						  const PeerVector &peerVector,
						  network_type::Type type,
						  bool isStreamer,
						  ::size_t me, ::size_t size,
						  ::size_t start, ::size_t size_2pow);

	void createConnectionDC(RWList &rwList,
							const PeerVector &peerVector,
							network_type::Type type,
							bool isStreamer,
							::size_t me, ::size_t size,
							::size_t start, ::size_t size_2pow);

	PeerVector peers;
	bool verbose;
};

inline std::ostream& operator<< (std::ostream& s, const Network &network) {
	return network.put(s);
}

}

}

#endif	// ndef MR_NET_NETIO_H__
