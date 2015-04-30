#include <sys/socket.h>
#include <netinet/in.h>

#include <string>
#include <vector>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#pragma GCC diagnostic pop

// #include <mr/util.h>
#include <mr/net/netio.h>


namespace mr {

/* borrowed from mr/util.h */
class Counter {
public:
	Counter() : n(0), size(0) {
	}

	void reset() {
		n = 0;
		size = 0;
	}

	void inc(long d) {
		n++;
		size += d;
	}

	std::ostream &put(std::ostream &s) const {
		return s << n << ":" << size;
	}

protected:
	int			n;
	long		size;
};


inline std::ostream& operator<< (std::ostream& s, const Counter &r) {
	return r.put(s);
}

namespace net {

class SocketNetworkException : public NetworkException {
public:
	SocketNetworkException(const std::string &reason)
			: NetworkException(reason + ": " + std::string(strerror(errno))) {
	}
};

class SockAddr {
public:
	static const ::size_t SOCKADDR_TEXT_SIZE = (8 * (4 + 1));

	virtual ~SockAddr() {
	}

	sa_family_t getFamily() const;

	::socklen_t getSize() const;

	virtual in_port_t getPort() const = 0;

	struct sockaddr *getAddr() const;

	void getSockName(int fd);

	virtual void setPort(in_port_t port) = 0;

	virtual bool isAny() const = 0;

	static SockAddr *createSockAddr(const struct sockaddr *a);

	static SockAddr *createSockAddr(sa_family_t family);

	static SockAddr *createWithInterface(const std::string &interface);

	static SockAddr *clone(const SockAddr &from);

	virtual std::ostream &put(std::ostream &str) const;

protected:
	SockAddr(sa_family_t family, ::socklen_t size, const sockaddr *a);

	sockaddr_storage addr;
	sa_family_t	family;
	::socklen_t	size;
	char name[SOCKADDR_TEXT_SIZE];
};

inline std::ostream &operator<< (std::ostream &str, const SockAddr &s) {
	return s.put(str);
}


class SockAddrIPv4 : public SockAddr {
public:
	SockAddrIPv4(const sockaddr *addr);

	virtual in_port_t getPort() const;

	virtual void setPort(in_port_t port);

	virtual bool isAny() const;
};


class SockAddrIPv6 : public SockAddr {
public:
	SockAddrIPv6(const sockaddr *addr);

	virtual in_port_t getPort() const;

	virtual void setPort(in_port_t port);

	virtual bool isAny() const;
};


namespace tcp {

class Socket {
public:
	static const std::string portSep;

	Socket(int fd = -1);

	virtual ~Socket();

	const SockAddr *getSockAddr() const;

#if 0
	::size_t getSockAddrSize() const;

	int getFamily();
#endif

	void closeReading();

	void closeWriting();

	int getFd();

	in_port_t getPort() const;

	void increaseSharing();

	static void discard(Socket *socket);

	virtual std::ostream &put(std::ostream &str) const;

protected:
	int fd;
	SockAddr *addr;
	int refCount;

private:
	void close();
};


class ClientSocket : public Socket {
public:
	// Wrapper about accepted socket. Takes ownership of @param peerAddr.
	ClientSocket(int fd, SockAddr *peerAddr);

	// Connect to ServerSocket designated by peerAddr
	ClientSocket(const SockAddr &peerAddr, const std::string &interface = "");

	virtual ~ClientSocket();

	// throws SocketNetworkException
	::size_t read(void *data, ::size_t count);

	::size_t read_fully(void *data, ::size_t count);

	// throws SocketNetworkException
	::size_t readv(const struct ::iovec *iov, int iovcnt);

	// throws ConnectionClosedException
	::size_t peek(void *buf, ::size_t len);

	// throws SocketNetworkException
	void write(const void *data, size_t count);

	// throws SocketNetworkException
	void writev(const struct iovec *iov_orig, int iovlen_orig);

	void setSocketOptions();

	virtual std::ostream &put(std::ostream &str) const;

protected:
	void connect(const SockAddr &peerAddr, const std::string &interface);

	SockAddr *peerAddr;

	static const ::size_t DEFAULT_BUFFER_SIZE = 1 << 20;
	void initBuffer();
	::size_t unBufferedRead(void *data, ::size_t count);
	::size_t unBufferedPeek(void *buf, ::size_t len);
	::size_t inputOffset;
	::size_t inputCount;
	::size_t inputCapacity;
	char *inputBuffer;
};


class ServerSocket : public Socket {

public:
	/**
	 * @param port if 0, let the system choose a suitable port
	 */
	ServerSocket(const std::string &interface, in_port_t port);

	virtual ~ServerSocket();

	ClientSocket *accept();

	const std::string &getName() const;

protected:
	std::string name;
};


}	// namespace tcp


class SocketNetwork;


class SocketNetworkPeer : public NetworkPeer {
public:
	// Takes ownership of @param addr
	SocketNetworkPeer(int peer, const SockAddr *addr);

#if 1
    virtual ~SocketNetworkPeer() {
		delete addr;
    }
#endif

	int getRank() const;

	const SockAddr &getAddr() const;

	virtual std::ostream &put(std::ostream &s) const {
		s << "Socket-" << peer << "-" << *addr << "-" << addr->getPort();

		return s;
	}

	virtual const std::string shortName() const {
		std::ostringstream o;
		o << peer;
		return o.str();
	}

protected:
	int		peer;
	const SockAddr *addr;
};


struct SocketReaderStats {
	Counter		recv;
	Counter		deliv;
};


struct SocketWriterStats {
	Counter		sent;
};


class SocketReader : public NetworkReader {

public:
	SocketReader(Network *network,
				 const SocketNetworkPeer &peer,
				 network_type::Type type,
				 bool isStreamer = false);

	virtual ~SocketReader();

	// throws NetworkException
	virtual ::size_t read(void *data, ::size_t count);

	// throws NetworkException
	virtual void readFully(void *data, ::size_t count);

	// throws NetworkException
	virtual ::size_t readv(const struct ::iovec *iov, int iovcnt);

	// return -1 when size cannot be determined
	virtual ::ssize_t getSize() const {
		return -1;
	}

	// throws SocketException
	virtual bool hasMore();

	virtual void close();

protected:
	tcp::ClientSocket *socket;

	SocketNetwork *socketNetwork;
	friend class SocketNetwork;

	SocketReaderStats stats;
};



class SocketWriter : public NetworkWriter {

public:
	SocketWriter(Network *network,
				 const SocketNetworkPeer &peer,
				 network_type::Type type,
				 bool isStreamer = false);

	virtual ~SocketWriter();

	virtual void write(const void *data, size_t count);

	virtual void writev(const struct iovec *iov, int iovlen);

	virtual void close();

protected:
	tcp::ClientSocket *socket;

	SocketNetwork *socketNetwork;

	SocketWriterStats stats;
};



/*-----------------------------------------------------------------------------
 *
 * class SocketImpl
 *
 *--------------------------------------------------------------------------- */

class SocketImpl : public NetworkImpl {
public:
	SocketImpl();

	virtual ~SocketImpl() {
	}

	virtual Network *createNetwork(const OptionList &options);

protected:
	int refCount;
	boost::mutex socketLock;
};

	
class PendingConnection {
public:
	PendingConnection() {
	}

	boost::condition_variable connectionArrived;
	tcp::ClientSocket *socket;
	PendingConnection *next;
};


class SocketMap {
public:
	typedef std::map<network_type::Type, PendingConnection *> TypeMap;

	SocketMap();

	SocketMap(int size);

	~SocketMap();

	PendingConnection *get(int peer, network_type::Type type);

	void release(PendingConnection *p);

protected:
	std::vector<TypeMap> pending;
	PendingConnection *freelist;
};


class SocketAcceptThread {
public:
	SocketAcceptThread(SocketNetwork &socketNetwork,
					   tcp::ServerSocket &serverSocket);

	virtual ~SocketAcceptThread() {
	}

	// boost speak for the run() method
	void operator() ();

	void stop();

protected:
	SocketNetwork &socketNetwork;
	tcp::ServerSocket &serverSocket;
};  


class ProbeInfo {
public:
	ProbeInfo() {
		FD_ZERO(&readfds);
		nfds = 0;
	}

	fd_set	readfds;
	int		nfds;
	std::map<int, SocketReader *> reader;
};


class SocketNetwork : public Network {

protected:
	typedef std::pair<tcp::ClientSocket *, network_type::Type> SocketTypePair;

public:
	SocketNetwork(const OptionList &options);

	virtual ~SocketNetwork();

	virtual ReaderInterface *createReader(const NetworkPeer &conn,
										  network_type::Type type,
										  bool isStreamer = false);

	virtual WriterInterface *createWriter(const NetworkPeer &conn,
										  network_type::Type type,
										  bool isStreamer = false);

	void unregisterReader(SocketReader *reader);

	void setInterfaceName(const std::string &interfaceName);

	const std::string &getInterfaceName() const;

	virtual const NetworkPeer *getMe() const;

	virtual std::ostream &put(std::ostream &s) const;

	virtual void probe(ReaderList &pending,
					   network_type::Type type,
					   bool block = true);

protected:
	// A SocketReader requests a pending connection from the incomingConnections
	// database
	tcp::ClientSocket *acceptFrom(const SocketNetworkPeer &conn,
								  network_type::Type type);
	// A SocketWriter requests a connection to some peer
	tcp::ClientSocket *connectTo(const SocketNetworkPeer &conn,
								 network_type::Type type);

	// The acceptThread registers a pending connection with the
	// incomingConnections database
	void registerConnection(int peer,
							network_type::Type type,
							tcp::ClientSocket *socket,
							bool isNew = true,
							int from = -1);

	void addProbeInfo(ProbeInfo *probeInfo, SocketReader *reader);
	void removeProbeInfo(ProbeInfo *probeInfo, SocketReader *reader);

	int			rank;
	int			size;

	std::string interfaceName;

	const NetworkPeer *me;

	SocketMap incomingConnections;
	boost::mutex connectionLock;
	boost::condition_variable connectionArrived;

	tcp::ServerSocket *serverSocket;
	SocketAcceptThread *accepter;
	boost::thread *acceptThread;

	std::map<network_type::Type, ProbeInfo> probeInfo;

	friend class SocketReader;
	friend class SocketWriter;
	friend class SocketAcceptThread;
};

}

}
