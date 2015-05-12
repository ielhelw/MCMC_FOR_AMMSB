
#include "netio_sockets.h"	// need .h file to break type hierarchy

#include <inttypes.h>

#include <sys/utsname.h>
#include <stdlib.h>

#include <sys/types.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <sys/time.h>
#include <sys/select.h>
#include <sys/uio.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <errno.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <das_inet_sync.h>

// #include <mr/util.h>
#include <mr/net/netio.h>
// #include <mr/log.h>


using std::vector;
using std::string;
using std::ostream;
using std::ostringstream;


namespace mr {

namespace net {


static bool isAny(const sockaddr_in *a) {
	return a->sin_addr.s_addr == INADDR_ANY;
}

static bool isAny(const sockaddr_in6 *a) {
	return memcmp(&a->sin6_addr.s6_addr, &in6addr_any, sizeof in6addr_any) == 0;
}

static bool isAny(const sockaddr *a) {
	switch (a->sa_family) {
	case AF_INET:
		return isAny((const sockaddr_in *)a);
	case AF_INET6:
		return isAny((const sockaddr_in6 *)a);
	}

	errno = EPROTONOSUPPORT;
	throw SocketNetworkException("Unknown/unsupported protocol");
}

/* --------------------------------------------------------------------------
 *
 * class SockAddr
 *
 * -------------------------------------------------------------------------- */

SockAddr::SockAddr(sa_family_t family, ::socklen_t size, const sockaddr *a)
		: family(family), size(size) {
	memcpy(&this->addr, a, size);
	if (mr::net::isAny(a)) {
		// Need this horrible hack, because getnameinfo() hits huge timeouts
		// for sockaddr ANY.
		snprintf(name, sizeof name, "<ANY>");
	} else {
		if (getnameinfo(a, size, name, (socklen_t)sizeof name, NULL, 0, 0) == -1) {
			throw SocketNetworkException("Cannot getnameinfo()");
		}
	}
}

sa_family_t SockAddr::getFamily() const {
	return family;
}

::socklen_t SockAddr::getSize() const {
	return size;
}

sockaddr *SockAddr::getAddr() const {
	return (struct sockaddr *)&addr;
}

void SockAddr::getSockName(int fd) {
	::socklen_t size = this->size;
	if (getsockname(fd, (struct sockaddr *)&addr, &size) == -1) {
		throw SocketNetworkException("Cannot getsockname");
	}
}


SockAddr *SockAddr::createSockAddr(sa_family_t family) {
	switch (family) {
	case AF_INET:
		struct sockaddr_in dummy4;
		// memset(&dummy4, 0, sizeof dummy4);
		dummy4.sin_family = AF_INET;
		dummy4.sin_addr.s_addr = INADDR_ANY;
		return new SockAddrIPv4((const sockaddr *)&dummy4);
	case AF_INET6:
		struct sockaddr_in6 dummy6;
		// memset(&dummy6, 0, sizeof dummy6);
		dummy6.sin6_family = AF_INET6;
		dummy6.sin6_addr = in6addr_any;
		return new SockAddrIPv6((const sockaddr *)&dummy6);
	default:
		errno = EAFNOSUPPORT;
		throw SocketNetworkException("Unknown/unsupported address family");
	}
}


SockAddr *SockAddr::createSockAddr(const struct sockaddr *a) {
	// switch (((const struct sockaddr_in *)a)->sin_family) {}
	switch (a->sa_family) {
	case AF_INET:
		return new SockAddrIPv4(a);
	case AF_INET6:
		return new SockAddrIPv6(a);
	default:
		errno = EAFNOSUPPORT;
		throw SocketNetworkException("Unknown/unsupported address family");
	}
}


SockAddr *SockAddr::createWithInterface(const string &interface) {
	SockAddr *addr = NULL;
	struct ifaddrs *ifap;
	const struct ifaddrs *ifa;

	if (getifaddrs(&ifap) == -1) {
		throw SocketNetworkException("Cannot getifaddrs()");
	}

	for (ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
		if (interface == "" || strcmp(interface.c_str(), ifa->ifa_name) == 0) {
			if (ifa->ifa_addr->sa_family == AF_INET ||
			   		ifa->ifa_addr->sa_family == AF_INET6) {
				addr = createSockAddr(ifa->ifa_addr);
				break;
			}
		}
	}

	freeifaddrs(ifap);

	if (addr == NULL) {
		errno = EPROTONOSUPPORT;
		throw SocketNetworkException("Interface not found");
	}

	return addr;
}


SockAddr *SockAddr::clone(const SockAddr &from) {
	switch (from.getFamily()) {
	case AF_INET:
		return new SockAddrIPv4(*(SockAddrIPv4 *)&from);
	case AF_INET6:
		return new SockAddrIPv6(*(SockAddrIPv6 *)&from);
	}

	errno = EAFNOSUPPORT;
	throw SocketNetworkException("Unknown/unsupported family");
}


std::ostream &SockAddr::put(std::ostream &str) const {
	str << name;

	return str;
}


/* --------------------------------------------------------------------------
 *
 * class SockAddrIPv4
 *
 * -------------------------------------------------------------------------- */

SockAddrIPv4::SockAddrIPv4(const struct sockaddr *addr)
   		: SockAddr((sa_family_t)AF_INET,
				   (::socklen_t)sizeof(struct sockaddr_in),
				   addr) {
}

in_port_t SockAddrIPv4::getPort() const {
	return ntohs(((struct sockaddr_in *)&addr)->sin_port);
}

void SockAddrIPv4::setPort(in_port_t port) {
	((sockaddr_in *)&addr)->sin_port = htons(port);
}

bool SockAddrIPv4::isAny() const {
	return mr::net::isAny((sockaddr_in *)&addr);
}


/* --------------------------------------------------------------------------
 *
 * class SockAddrIPv6
 *
 * -------------------------------------------------------------------------- */

SockAddrIPv6::SockAddrIPv6(const struct sockaddr *addr)
   		: SockAddr((sa_family_t)AF_INET6,
				   (::socklen_t)sizeof(struct sockaddr_in6),
				   addr) {
}

in_port_t SockAddrIPv6::getPort() const {
	return ntohs(((struct sockaddr_in6 *)&addr)->sin6_port);
}

void SockAddrIPv6::setPort(in_port_t port) {
	((sockaddr_in6 *)&addr)->sin6_port = htons(port);
}

bool SockAddrIPv6::isAny() const {
	return mr::net::isAny((sockaddr_in6 *)&addr);
}


::size_t IPv6_TEXT_SIZE;


namespace tcp {


/* --------------------------------------------------------------------------
 *
 * class Socket
 *
 * -------------------------------------------------------------------------- */


Socket::Socket(int fd) : fd(fd), addr(NULL), refCount(1) {
}

Socket::~Socket() {
	close();
	delete addr;
}

const SockAddr *Socket::getSockAddr() const {
	return addr;
}

#if 0
::size_t Socket::getSockAddrSize() const {
	return addr->getSize();
}

int Socket::getFamily() {
	return addr->getFamily();
}
#endif

void Socket::closeReading() {
	::shutdown(fd, SHUT_RD);
	close();
}

void Socket::closeWriting() {
	::shutdown(fd, SHUT_WR);
	close();
}

int Socket::getFd() {
	return fd;
}

in_port_t Socket::getPort() const {
	return addr->getPort();
}

void Socket::increaseSharing() {
	refCount++;
	// std::cerr << "socket " << this << " up refCount now " << refCount << std::endl;
}

void Socket::discard(Socket *socket) {
	if (socket->refCount == 0) {
		delete socket;
	}
}

std::ostream &Socket::put(std::ostream &str) const {
	str << "fd " << fd << " " << *addr;

	return str;
}

void Socket::close() {
	if (fd != -1) {
		refCount--;
		// std::cerr << "socket " << this << " close(), refCount now " << refCount << std::endl;
		if (refCount == 0) {
			// std::cerr << "close socket fd " << fd << std::endl;
			::shutdown(fd, SHUT_RDWR);
			::close(fd);
			fd = -1;
		}
	}
}

const string Socket::portSep = "/";

std::ostream &operator<< (std::ostream &str, const Socket &s) {
	return s.put(str);
}



/* --------------------------------------------------------------------------
 *
 * class ClientSocket
 *
 * -------------------------------------------------------------------------- */

const ::size_t ClientSocket::DEFAULT_BUFFER_SIZE;

// Wrapper about accepted socket. Takes ownership of @param peerAddr.
ClientSocket::ClientSocket(int fd, SockAddr *peerAddr)
		: Socket(fd), peerAddr(peerAddr) {
	initBuffer();
	addr = SockAddr::createSockAddr(peerAddr->getFamily());
	addr->getSockName(fd);

	setSocketOptions();
}


ClientSocket::ClientSocket(const SockAddr &peerAddr, const string &interface) {
	initBuffer();
	try {
		if (interface != "") {
			addr = SockAddr::createWithInterface(interface);
			if (addr->getFamily() != peerAddr.getFamily()) {
				errno = EPROTO;
				throw SocketNetworkException("family mismatch");
			}
		} else {
			addr = SockAddr::createSockAddr(peerAddr.getFamily());
		}
		connect(peerAddr, interface);
	} catch (SocketNetworkException &e) {
		if (fd != -1) {
			::close(fd);
		}

		throw e;
	}
}


ClientSocket::~ClientSocket() {
	delete peerAddr;
	delete[] inputBuffer;
}


void ClientSocket::initBuffer() {
	inputOffset = 0;
	inputCount  = 0;
	inputCapacity = DEFAULT_BUFFER_SIZE;
	inputBuffer = new char[inputCapacity];
	// std::cerr << "Accepted socket: " << *addr << ":" << addr->getPort() << std::endl;
	// std::cerr << "create socket " << this << " refCount " << refCount << std::endl;
}


// throws SocketNetworkException
::size_t ClientSocket::unBufferedRead(void *data, ::size_t count) {
	while (true) {
		// std::cerr << "Wanna read " << count << std::endl;
		::ssize_t r = ::read(fd, data, count);
		if (r == -1) {
			if (errno == EINTR) {
				errno = 0;
			} else {
				std::ostringstream o;
				o << "Cannot read from socket: errno=" << errno << ", msg=" << strerror(errno);
				std::cerr << o.str() << std::endl;
				throw SocketNetworkException(o.str());
			}
		} else {
			// std::cerr << "Read " << r << " checksum " << checksum(data, r) << std::endl;
			return static_cast< ::size_t>(r);
		}
	}
}

::size_t ClientSocket::read(void *data, ::size_t count) {
	if (inputCount == 0) {
		::size_t r = unBufferedRead(inputBuffer, inputCapacity);
		inputCount = r;
	}

	count = std::min(count, inputCount);
	memcpy(data, inputBuffer + inputOffset, count);
	inputCount -= count;
	inputOffset += count;
	if (inputCount == 0) {
		inputOffset = 0;
	}

	return count;
}

::size_t ClientSocket::read_fully(void *data, ::size_t count) {
	::size_t total = 0;
	while (total < count) {
		::size_t rd = read((char *)data + total, count - total);
		if (rd == 0) {
			throw ConnectionClosedException("read_fully cannot comply");
		}
		total += rd;
	}

	return total;
}


// throws SocketNetworkException
::size_t ClientSocket::readv(const struct ::iovec *iov, int iovcnt) {
	throw SocketNetworkException("Not implemented w/ buffering");
	while (true) {
		::ssize_t r = ::readv(fd, iov, iovcnt);
		if (r == -1) {
			if (errno == EINTR) {
				errno = 0;
			} else {
				throw SocketNetworkException("Cannot read from socket");
			}
		} else {
			return static_cast< ::size_t>(r);
		}
	}
}

// throws ConnectionClosedException
::size_t ClientSocket::unBufferedPeek(void *buf, ::size_t len) {
	::ssize_t r = recv(fd, buf, len, MSG_PEEK);
	if (r == -1) {
		std::ostringstream o;
		o << "Connection error ClientSocket::" << __func__ << strerror(errno);
		std::cerr << o.str() << std::endl;
		throw SocketNetworkException(o.str());
	}

	return r;
}

// throws ConnectionClosedException
::size_t ClientSocket::peek(void *buf, ::size_t len) {
	if (inputCount > 0) {
		return std::min(len, inputCount);
	}

	return unBufferedPeek(buf, len);
}


// throws SocketNetworkException
void ClientSocket::write(const void *data, size_t count) {
	::size_t total = 0;
	while (total < count) {
		::ssize_t w = ::write(fd, (const char *)data + total, count - total);
		if (w == -1) {
			if (errno == EINTR) {
				std::cerr << "ClientSocket::" << __func__ << "(): Something interrupted me" << std::endl;
				errno = 0;
			} else {
				std::ostringstream o;
				o << "Cannot write: errno=" << errno << ", msg=" << strerror(errno);
				std::cerr << o.str() << std::endl;
				throw SocketNetworkException(o.str());
			}
		} else {
			if ((::size_t)w != count - total) {
				std::cerr << "ClientSocket::" << __func__ << "(" << (count - total) << ") -> " << w << std::endl;
			}
			total += w;
		}
	}
}


// throws SocketNetworkException
void ClientSocket::writev(const struct iovec *iov_orig, int iovlen_orig) {
	int iovlen = iovlen_orig;
	struct iovec iov[iovlen];
	::size_t count = 0;
	for (int i = 0; i < iovlen; i++) {
		iov[i] = iov_orig[i];
		count += iov[i].iov_len;
	}

	::size_t total = 0;
	struct iovec *iov_current = iov;
	while (total < count) {
		::ssize_t w = ::writev(fd, iov_current, iovlen);
		if (w == -1) {
			if (errno == EINTR) {
				errno = 0;
			} else {
				throw SocketNetworkException("Cannot writev");
			}
		} else {
			total += w;
			while (true) {
				if ((::ssize_t)iov_current[0].iov_len > w) {
					iov_current[0].iov_len -= w;
					iov_current[0].iov_base = (char *)iov_current[0].iov_base + w;
					break;
				} else {
					w -= iov_current[0].iov_len;
					iov_current++;
				}
			}
		}
	}
}

std::ostream &ClientSocket::put(std::ostream &str) const {
	str << "fd " << fd << " me " << *addr << ":" << addr->getPort() << " peer " << *peerAddr << ":" << peerAddr->getPort();

	return str;
}


void ClientSocket::setSocketOptions() {
	int one = 1;
	if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, (socklen_t)sizeof one) == -1) {
		throw SocketNetworkException("Cannot setsockopt(TCP_NODELAY)");
	}
	if (false) {
		socklen_t buffersize = DEFAULT_BUFFER_SIZE;
		std::cerr << "Set socket buffer size to " << DEFAULT_BUFFER_SIZE << std::endl;
		if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffersize, (socklen_t)sizeof buffersize) == -1) {
			throw SocketNetworkException("Cannot setsockopt(SO_RCVBUF)");
		}
		if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffersize, (socklen_t)sizeof buffersize) == -1) {
			throw SocketNetworkException("Cannot setsockopt(SO_RCVBUF)");
		}
	}
}


void ClientSocket::connect(const SockAddr &peerAddr, const std::string &interface) {
	fd = socket(peerAddr.getFamily(), SOCK_STREAM, 0);
	if (fd == -1) {
		throw SocketNetworkException("Cannot create socket");
	}

	setSocketOptions();

	// std::cerr << "Before bind: " << *addr << " port " << addr->getPort() << std::endl;
	addr->setPort(0);
	if (interface != "") {
		if (false) {
			int level;
			switch (peerAddr.getFamily()) {
			case AF_INET:
				level = SOL_SOCKET;
				break;
			case AF_INET6:
				level = IPPROTO_IPV6;
				break;
			}

			struct ifreq ifreq;
			strncpy(ifreq.ifr_ifrn.ifrn_name, interface.c_str(), IFNAMSIZ);

			if (setsockopt(fd, level, SO_BINDTODEVICE, &ifreq, (socklen_t)sizeof ifreq) == -1) {
				throw SocketNetworkException("Cannot setsockopt(BINDTODEVICE)");
			}
		} else {
			if (bind(fd, addr->getAddr(), addr->getSize()) == -1) {
				throw SocketNetworkException("Cannot bind socket");
			}
		}
		addr->getSockName(fd);
		// std::cerr << "After bind: " << *addr << " port " << addr->getPort() << std::endl;
	}

	if (0) {
		struct ifreq ifr;
		ifr.ifr_ifindex = ((struct sockaddr_in6 *)addr->getAddr())->sin6_scope_id;
		if (ioctl(fd, SIOCGIFNAME, &ifr) == 01) {
			throw SocketNetworkException("Cannot ioctl(SIOCGIFNAME)");
		}
		std::cerr << "Interface name[" << ((struct sockaddr_in6 *)addr->getAddr())->sin6_scope_id << "] " << ifr.ifr_name << std::endl;
		std::cerr << "Connect to peer " << peerAddr << " port " << peerAddr.getPort() << std::endl;
	}

	while (::connect(fd, peerAddr.getAddr(), peerAddr.getSize()) == -1) {
		if (errno == EINTR) {
			errno = 0;
		} else {
			throw SocketNetworkException("Cannot connect socket");
		}
	}

	addr->getSockName(fd);

	this->peerAddr = SockAddr::clone(peerAddr);

	if (false) {
		std::cerr << "Established connection: fd " << fd << " me " << *addr << ":" << addr->getPort() << " peer " << peerAddr << ":" << peerAddr.getPort() << std::endl;
	}

}

SockAddr *peerAddr;



/* --------------------------------------------------------------------------
 *
 * class ServerSocket
 *
 * -------------------------------------------------------------------------- */


/**
 * @param port if 0, let the system choose a suitable port
 */
ServerSocket::ServerSocket(const string &interface, in_port_t port)
		: Socket(-1) {
	try {
		addr = SockAddr::createWithInterface(interface);
		fd = socket(addr->getFamily(), SOCK_STREAM, 0);
		if (fd == -1) {
			throw SocketNetworkException("Cannot create socket");
		}
		addr->setPort(port);
		if (bind(fd, addr->getAddr(), addr->getSize()) == -1) {
			throw SocketNetworkException("Cannot bind socket");
		}

		int one = 1;
		if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, (socklen_t)sizeof one) == -1) {
			throw SocketNetworkException("Cannot setsockopt(REUSEADDR)");
		}
		if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, (socklen_t)sizeof one) == -1) {
			throw SocketNetworkException("Cannot setsockopt(TCP_NODELAY)");
		}
		if (setsockopt(fd, IPPROTO_TCP, SO_KEEPALIVE, &one, (socklen_t)sizeof one) == -1) {
			throw SocketNetworkException("Cannot setsockopt(TCP_NODELAY)");
		}

		struct linger linger = { 1, 10 };
		if (setsockopt(fd, SOL_SOCKET, SO_LINGER, &linger, (socklen_t)sizeof linger) == -1) {
			throw SocketNetworkException("Cannot setsockopt(LINGER)");
		}

		addr->getSockName(fd);
		// std::cerr << "Create ServerSocket " << *addr << ":" << addr->getPort() << std::endl;

		ostringstream o;
		if (addr->isAny()) {
			struct utsname uts;

			if (uname(&uts) == -1) {
				throw SocketNetworkException("Cannot uname()");
			}
			o << uts.nodename << portSep << addr->getPort();
		} else {
			o << *addr << portSep << addr->getPort();
		}
		name = o.str();

		if (listen(fd, 5) != 0) {
			throw SocketNetworkException("Cannot listen socket");
		}

	} catch (SocketNetworkException &e) {
		if (fd != -1) {
			::close(fd);
			fd = -1;
		}

		throw e;
	}
}


ServerSocket::~ServerSocket() {
}


ClientSocket *ServerSocket::accept() {
	socklen_t   size;
	sockaddr_storage peerAddr;
	int         clientFd = -1;

	size = (socklen_t)sizeof peerAddr;
	do {
		clientFd = ::accept(fd, (struct sockaddr *)&peerAddr, &size);
		if (clientFd == -1) {
			if (errno != EINTR) {
				throw SocketNetworkException("Cannot accept socket");
			}
			errno = 0;
		}
	} while (clientFd == -1);

	// the new ClientSocket becomes owner of the created SockAddr:
	SockAddr *p = SockAddr::createSockAddr((struct sockaddr *)&peerAddr);
	ClientSocket *s = new ClientSocket(clientFd, p);
	s->setSocketOptions();

	if (false) {
		std::cerr << "Accept connection: " << *s << std::endl;
	}

	return s;
}


const std::string &ServerSocket::getName() const {
	return name;
}

}	// namespace tcp



/* --------------------------------------------------------------------------
 *
 * class SocketNetworkPeer
 *
 * -------------------------------------------------------------------------- */

SocketNetworkPeer::SocketNetworkPeer(int peer, const SockAddr *addr)
		: peer(peer), addr(addr) {
}

int SocketNetworkPeer::getRank() const {
	return peer;
}

const SockAddr &SocketNetworkPeer::getAddr() const {
	return *addr;
}


/* --------------------------------------------------------------------------
 *
 * class SocketReader
 *
 * -------------------------------------------------------------------------- */

SocketReader::SocketReader(Network *network,
						   const SocketNetworkPeer &peer,
						   network_type::Type type,
						   bool isStreamer)
		: NetworkReader(network, peer, type, isStreamer),
		  socketNetwork(dynamic_cast<SocketNetwork *>(network)) {
	socket = socketNetwork->acceptFrom(peer, type);
}


SocketReader::~SocketReader() {
	close();
}


// throws NetworkException
::size_t SocketReader::read(void *data, ::size_t count) {
	if (false && socketNetwork->verbose) {
		std::cerr << "socket " << *socket->getSockAddr() << " fd " << socket->getFd() << " try to " << __func__ << "(" << count << ")" << std::endl;
	}
	::size_t r = socket->read(data, count);
	if (socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " " << __func__ << "(" << r << ")" << std::endl;
	}

	stats.recv.inc(r);
	stats.deliv.inc(r);

	return r;
}


// throws NetworkException
void SocketReader::readFully(void *data, ::size_t count) {
	if (false && socketNetwork->verbose) {
		std::cerr << "socket " << *socket->getSockAddr() << " fd " << socket->getFd() << " try to " << __func__ << "(" << count << ")" << std::endl;
	}
	socket->read_fully(data, count);
	if (socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " " << __func__ << "(" << count << ")" << std::endl;
	}

	stats.recv.inc(count);
	stats.deliv.inc(count);
}


// throws NetworkException
::size_t SocketReader::readv(const struct ::iovec *iov, int iovcnt) {
	if (false && socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " try to " << __func__ << "()" << std::endl;
	}
	::size_t r = socket->readv(iov, iovcnt);
	if (socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " " << __func__ << "(" << r << ")" << std::endl;
	}

	stats.recv.inc(r);
	stats.deliv.inc(r);

	return r;
}


// throws SocketException
// There is only one way to be sure whether there will be more: wait until
// there is a message or the connection is closed.
bool SocketReader::hasMore() {
	try {
		int32_t	dummy;
		::size_t r = socket->peek(&dummy, sizeof dummy);
	   	return (r != 0);
	} catch (ConnectionClosedException &e) {
		(void)e;
		return false;
	}
}

void SocketReader::close() {
	if (socket != NULL) {
		if (socketNetwork->verbose) {
			std::cerr << "Close reader type " << type << " socket " << *socket->getSockAddr() << "(peer=" << peer << ") w/ fd " << socket->getFd() << std::endl;
		}
		socketNetwork->unregisterReader(this);
		socket->closeReading();
		tcp::Socket::discard(socket);
		socket = NULL;
		std::cerr << "Reader(peer=" << peer << ",type=" << type << ") recv " << stats.recv << " deliv " << stats.deliv << std::endl;
	}
}




/* --------------------------------------------------------------------------
 *
 * class SocketWriter
 *
 * -------------------------------------------------------------------------- */

SocketWriter::SocketWriter(Network *network,
						   const SocketNetworkPeer &peer,
						   network_type::Type type,
						   bool isStreamer)
		: NetworkWriter(network, peer, type, isStreamer),
		  socketNetwork(dynamic_cast<SocketNetwork *>(network)) {
	socket = socketNetwork->connectTo(peer, type);
}

SocketWriter::~SocketWriter() {
	close();
}

void SocketWriter::write(const void *data, size_t count) {
	// std::cerr << "Write to " << peer << " size " << count << " checksum " << checksum(data, count) << std::endl;
	socket->write(data, count);
	if (socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " " << __func__ << "(" << count << ")" << std::endl;
	}
	// std::cerr << "Wrote to " << peer << " size " << count << " checksum " << checksum(data, count) << std::endl;

	stats.sent.inc(count);
}

void SocketWriter::writev(const struct iovec *iov, int iovlen) {
	socket->writev(iov, iovlen);
	::size_t r = 0;
	for (int i = 0; i < iovlen; i++) {
		r += iov[i].iov_len;
	}

	if (socketNetwork->verbose) {
		std::cerr << "socket " << socket << " fd " << socket->getFd() << " " << __func__ << "(" << r << ")" << std::endl;
	}

	stats.sent.inc(r);
}

void SocketWriter::close() {
	if (socket != NULL) {
		if (socketNetwork->verbose) {
			std::cerr << "Close writer type " << type << " socket " << *socket->getSockAddr() << "(peer=" << peer << ") w/fd " << socket->getFd() << std::endl;
		}
if (0) {
	std::cerr << "Sleep for 3 seconds...." << std::endl;
struct timespec dt = { 3, 0 };
nanosleep(&dt, NULL);
}

		socket->closeWriting();

		std::cerr << "Writer(peer=" << peer << ",type=" << type << ") sent " << stats.sent << std::endl;

		tcp::Socket::discard(socket);
		socket = NULL;
	}
}


namespace ConnectionTags {
	enum Tag {
		NEW,
		STOP,
		PAIR
	};
};


/*-----------------------------------------------------------------------------
 *
 * class ConnectionTag
 *
 *--------------------------------------------------------------------------- */

struct ConnectionTag {
	ConnectionTag() {
	}

	ConnectionTag(ConnectionTags::Tag tag) : tag((int32_t)tag), type(0), from(0), pair(0) {
	}

	ConnectionTag(ConnectionTags::Tag tag, network_type::Type type, int from)
			: tag((int32_t)tag), type((int32_t)type), from(from), pair(0) {
	}

	ConnectionTag(ConnectionTags::Tag tag, network_type::Type type, int from, int pair)
			: tag((int32_t)tag), type((int32_t)type), from(from), pair(pair) {
	}

	int32_t		tag;
	int32_t		type;
	int32_t		from;
	int32_t		pair;
};


/*-----------------------------------------------------------------------------
 *
 * class SocketAcceptThread
 *
 *--------------------------------------------------------------------------- */

SocketAcceptThread::SocketAcceptThread(SocketNetwork &socketNetwork,
									   tcp::ServerSocket &serverSocket)
		: socketNetwork(socketNetwork), serverSocket(serverSocket) {
}


// boost speak for the run() method
void SocketAcceptThread::operator() () {
	while (true) {
		tcp::ClientSocket *c = serverSocket.accept();
		ConnectionTag t;
		c->read_fully(&t, sizeof t);
		switch ((ConnectionTags::Tag)t.tag) {
		case ConnectionTags::STOP:
            delete c;
			return;
		case ConnectionTags::NEW:
			socketNetwork.registerConnection(t.from,
											 (network_type::Type)t.type,
											 c);
			break;
		case ConnectionTags::PAIR:
			if (socketNetwork.verbose) {
				std::cerr << "Receive PAIR request from peer " << t.from << " for tag " << t.pair << std::endl;
			}
			socketNetwork.registerConnection(t.from,
											 (network_type::Type)t.type,
											 c,
											 false,
											 t.pair);
			break;
		default:
			std::cerr << "Unknown connection tag " << t.tag << ": ignore" << std::endl;
			break;
		}
	}
}


void SocketAcceptThread::stop() {
	ostringstream s;
	tcp::ClientSocket c(*serverSocket.getSockAddr());
	ConnectionTag t(ConnectionTags::STOP);
	c.write(&t, sizeof t);
}


/* --------------------------------------------------------------------------
 *
 * class SocketMap
 *
 * -------------------------------------------------------------------------- */

SocketMap::SocketMap() : freelist(NULL) {
}   


SocketMap::SocketMap(int size) : pending(size), freelist(NULL) {
}   


SocketMap::~SocketMap() {
	PendingConnection *p;
	while ((p = freelist) != NULL) {
		freelist = p->next;
		delete p;
	}
}   


// Caller must lock this SocketMap
PendingConnection *SocketMap::get(int peer, network_type::Type type) {
	PendingConnection *c;
	TypeMap::iterator p = pending[peer].find(type);
	if (p != pending[peer].end()) {
		// some accepter already waiting for us, return the SocketMap entry
		c = p->second;
		pending[peer].erase(p);
	} else {
		// no accepter waiting, create a PendingConnection entry for it
		c = freelist;
		if (c == NULL) {
			c = new PendingConnection();
		} else {
			freelist = freelist->next;
		}
		c->socket = NULL;
		pending[peer][type] = c;
	}

	return c;
}


// Caller must lock this SocketMap
void SocketMap::release(PendingConnection *p) {
	p->next = freelist;
	freelist = p;
}


/*-----------------------------------------------------------------------------
 *
 * class SocketImpl
 *
 *--------------------------------------------------------------------------- */

#ifdef DONT_USE_STATIC_INITIALIZER
SocketImpl::SocketImpl() : NetworkImpl("socket"), refCount(0) {
	addNetworkImpl(this);
}


Network *SocketImpl::createNetwork(const OptionList &options) {
	return new SocketNetwork(options);
}


static SocketImpl impl;
#endif


/* --------------------------------------------------------------------------
 *
 * class SocketNetwork
 *
 * -------------------------------------------------------------------------- */

/*
 * Setup the basic socket connections: publish our listening socket(s).
 * Easy bootstrap solution: use the daslib sync server.
 */
SocketNetwork::SocketNetwork(const OptionList &options) {
	for (::size_t i = 0; i < options.size(); i++) {
		if (false) {
		} else if (options[i].first == "verbose") {
			verbose = (options[i].second == "on");
		} else if (options[i].first == "interface") {
			interfaceName = options[i].second;
		} else {
			std::cerr << "Unknown SocketNetwork option: \"" <<
			   	options[i].first << "=" << options[i].second << "\"" << std::endl;
		}
	}

	const char *rankEnv = getenv("PRUN_CPU_RANK");
	if (rankEnv == NULL) {
		errno = EINVAL;
		throw SocketNetworkException("Environment variable PRUN_CPU_RANK not defined");
	}
	if (sscanf(rankEnv, "%d", &rank) != 1) {
		errno = EINVAL;
		throw SocketNetworkException("Environment variable PRUN_CPU_RANK must be numeric");
	}

	const char *hostlistEnv = getenv("PRUN_HOSTNAMES");
	if (hostlistEnv == NULL) {
		errno = EINVAL;
		throw SocketNetworkException("Environment variable PRUN_HOSTNAMES not defined");
	}
	string hosts(hostlistEnv);
	boost::trim(hosts);
	vector<string> hostVector;
	boost::split(hostVector, hosts, boost::is_any_of(" "), boost::token_compress_on);
	size = static_cast<int>(hostVector.size());
	if (verbose) {
		std::cerr << "From PRUN_HOSTNAMES I find:" << std::endl;
		std::cerr << "   PRUN_HOSTNAMES \"" << hostlistEnv << "\"" << std::endl;
		for (::size_t i = 0; i < hostVector.size(); i++) {
			std::cerr << "    " << hostVector[i] << std::endl;
		}
	}

	incomingConnections = SocketMap(size);

	serverSocket = new tcp::ServerSocket(interfaceName, (in_port_t)0);

	int argc = 0;
	char *argv[] = { NULL };
	if (das_inet_sync_init(&argc, argv) == -1) {
		throw SocketNetworkException("Cannot das_inet_sync_init");
	}
	if (das_inet_sync_send(rank, size,
						   serverSocket->getSockAddr()->getAddr(),
						   serverSocket->getSockAddr()->getSize(),
						   "GlassWing") == -1) {
		throw SocketNetworkException("Cannot das_inet_sync_send");
	}
	sockaddr_storage remote_addr[size];
	if (das_inet_sync_rcve(SOCK_STREAM, remote_addr, size) == -1) {
		throw SocketNetworkException("Cannot das_inet_sync_rcve");
	}

	for (int i = 0; i < size; i++) {
		SockAddr *addr = SockAddr::createSockAddr((struct sockaddr *)&remote_addr[i]);
		SocketNetworkPeer *p = new SocketNetworkPeer((int)i, addr);
		if (i == rank) {
			me = p;
		}
		peers.push_back(p);
	}

	accepter = new SocketAcceptThread(*this, *serverSocket);
	// start daemon to listen to serverSocket
	acceptThread = new boost::thread(*accepter);
}


SocketNetwork::~SocketNetwork() {
	accepter->stop();
	acceptThread->join();

	delete acceptThread;
	delete accepter;
	delete serverSocket;

	while (! peers.empty()) {
		delete *peers.begin();
		peers.erase(peers.begin());
	}
}


void SocketNetwork::addProbeInfo(ProbeInfo *probeInfo, SocketReader *reader) {
	int		fd = reader->socket->getFd();
	FD_SET(fd, &probeInfo->readfds);
	probeInfo->reader[fd] = reader;
	if (fd >= probeInfo->nfds) {
		probeInfo->nfds = fd + 1;
	}
}


void SocketNetwork::removeProbeInfo(ProbeInfo *probeInfo,
								   	SocketReader *reader) {
	int		fd = reader->socket->getFd();
	FD_CLR(fd, &probeInfo->readfds);
	probeInfo->reader.erase(fd);
	if (fd + 1 == probeInfo->nfds) {
		for (int i = fd - 1; i >= 0; i--) {
			if (FD_ISSET(i, &probeInfo->readfds)) {
				probeInfo->nfds = i + 1;
				break;
			}
		}
	}
}


void SocketNetwork::unregisterReader(SocketReader *reader) {
	removeProbeInfo(&probeInfo[reader->type], reader);
	removeProbeInfo(&probeInfo[network_type::ANY], reader);
}


#if 0

class InputBuffer : public NetworkReader {
public:
	// Takes ownership of @param reader; @param reader will be deleted
	// when this is deleted.
	InputBuffer(NetworkReader *reader, ::size_t capacity)
			: NetworkReader(reader->network,
						   	reader->peer,
						   	reader->type,
						   	reader->isStreamer),
		   	  reader(reader), buf(new Buffer(capacity)) {
	}

	virtual ~InputBuffer() {
		close();
		delete buf;
	}

	virtual ::size_t read(void *ptr, ::size_t size) {
		assert(reader != NULL);

		std::cerr << __LINE__ << ": InputBuffer::" << __func__ << "(): count " << buf->getCount() << " capacity " << buf->getCapacity() << std::endl;
		if (buf->getCount() == 0) {
			buf->readFrom(*reader, buf->getCapacity());
			std::cerr << __LINE__ << ": InputBuffer::" << __func__ << "(): count " << buf->getCount() << " capacity " << buf->getCapacity() << std::endl;
		}

		size = std::min(size, buf->getCount());
		buf->readFrom(ptr, size);

		return size;
	}

	virtual ::size_t readv(const struct ::iovec *iov, int iovlen) {
		assert(reader != NULL);

		std::cerr << __LINE__ << ": InputBuffer::" << __func__ << "(): count " << buf->getCount() << " capacity " << buf->getCapacity() << std::endl;
		if (buf->getCount() == 0) {
			buf->readFrom(*reader, buf->getCapacity());
			std::cerr << __LINE__ << ": InputBuffer::" << __func__ << "(): count " << buf->getCount() << " capacity " << buf->getCapacity() << std::endl;
		}

		::size_t size = 0;
		for (int i = 0; i < iovlen; i++) {
			size += iov[i].iov_len;
		}
		size = std::min(size, buf->getCount());
		::size_t rd = size;
		int i = 0;
		while (rd > 0) {
			::size_t eSize = std::min(rd, iov[i].iov_len);
			buf->readFrom(iov[i].iov_base, eSize);
			rd -= eSize;
			i++;
		}

		return size;
	}

	virtual bool hasMore() {
		if (buf->getCount() > 0) {
			return true;
		}

		return reader->hasMore();
	}

	virtual void close() {
		if (reader != NULL) {
			delete reader;
			reader = NULL;
		}
	}

	virtual ::ssize_t getSize() const {
		return -1;
	}

protected:
	NetworkReader *reader;
	Buffer *buf;
};

#endif


ReaderInterface *SocketNetwork::createReader(const NetworkPeer &conn,
											 network_type::Type type,
											 bool isStreamer) {
	SocketReader *reader;
	reader = new SocketReader(this,
							  dynamic_cast<const SocketNetworkPeer &>(conn),
							  type,
							  isStreamer);

	addProbeInfo(&probeInfo[type], reader);
	addProbeInfo(&probeInfo[network_type::ANY], reader);

	return reader;
}


WriterInterface *SocketNetwork::createWriter(const NetworkPeer &conn,
											 network_type::Type type,
											 bool isStreamer) {
	WriterInterface *w;

	w = new SocketWriter(this,
						 dynamic_cast<const SocketNetworkPeer &>(conn),
						 type,
						 isStreamer);

	return w;
}


void SocketNetwork::setInterfaceName(const std::string &interfaceName) {
	this->interfaceName = interfaceName;
}


const std::string &SocketNetwork::getInterfaceName() const {
	return interfaceName;
}


const NetworkPeer *SocketNetwork::getMe() const {
	return me;
}


void SocketNetwork::probe(Network::ReaderList &pending,
						  network_type::Type type,
						  bool block) {
	struct timeval *timeout;
	struct timeval t0 = { 0, 0 };
	if (block) {
		timeout = NULL;
	} else {
		timeout = &t0;
	}

	fd_set readfds = probeInfo[type].readfds;
	fd_set exceptfds = probeInfo[type].readfds;
	int nfds = probeInfo[type].nfds;

	if (false) {
		std::ostringstream o;
		o << "Select on fds ";
		for (int i = 0; i < nfds; i++) {
			if (FD_ISSET(i, &readfds)) {
				o << i << ", ";
			}
		}
		std::cerr << o.str() << std::endl;
	}

	if (select(nfds, &readfds, NULL, &exceptfds, timeout) == -1) {
		throw SocketNetworkException("select() fails");
	}

	pending.clear();
	for (int i = 0; i < nfds; i++) {
		if (FD_ISSET(i, &readfds)) {
			pending.push_back(probeInfo[type].reader[i]);
		}
		if (FD_ISSET(i, &exceptfds)) {
			std::cerr << "See exception on socket " << i << std::endl;
		}
	}
}


tcp::ClientSocket *SocketNetwork::acceptFrom(const SocketNetworkPeer &peer,
											 network_type::Type type) {
	boost::mutex::scoped_lock lock(connectionLock);

	PendingConnection *p = incomingConnections.get(peer.getRank(), type);
	while (p->socket == NULL) {
		p->connectionArrived.wait(lock);
	}
	tcp::ClientSocket *socket = p->socket;
	incomingConnections.release(p);

	if (verbose) {
		std::cerr << "Retrieve accepted socket " << socket << " " << *socket << std::endl;
	}

	return socket;
}


tcp::ClientSocket *SocketNetwork::connectTo(const SocketNetworkPeer &conn,
											network_type::Type type) {
	boost::mutex::scoped_lock lock(connectionLock);

	tcp::ClientSocket *socket;

	socket = new tcp::ClientSocket(conn.getAddr(), interfaceName);
	ConnectionTag t(ConnectionTags::NEW, type, rank);
	socket->write(&t, sizeof t);

	return socket;
}


void SocketNetwork::registerConnection(int peer,
									   network_type::Type type,
									   tcp::ClientSocket *socket,
									   bool isNew,
									   int from) {
	boost::mutex::scoped_lock lock(connectionLock);

	PendingConnection *p = incomingConnections.get(peer, type);
	p->socket = socket;
	p->connectionArrived.notify_one();
	if (verbose) {
		std::cerr << "Register accepted socket " << socket << " " << *socket << " isNew " << isNew << " from " << from << std::endl;
	}
}


std::ostream &SocketNetwork::put(std::ostream &s) const {
	s << "socket network, rank " << rank << " of " << size;
	return s;
}

}

}
