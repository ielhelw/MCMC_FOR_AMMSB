/*
 * io.h
 *
 *  Created on: Mar 6, 2012
 *      Author: ielhelw
 */

#ifndef MR_IO_H_
#define MR_IO_H_

#include <sys/uio.h>	// struct iovec

#include <boost/thread/mutex.hpp>

#include "mcmc/exception.h"

namespace mr {


class ReaderInterface {
public:
	virtual ~ReaderInterface() {
	}

	/**
	 * @arg count in bytes
	 * @return bytes
	 * @throws IOException
	 */
	virtual ::size_t read(void *data, ::size_t count) = 0;

	/**
	 * @arg count in bytes
	 * @return bytes
	 * @throws IOException
	 */
	virtual ::size_t readv(const struct ::iovec *iov, int iovcnt) {
		::size_t total = 0;
		for (int i = 0; i < iovcnt; i++) {
			::size_t rd = read(iov[i].iov_base, iov[i].iov_len);
			total += rd;
			if (rd != iov[i].iov_len) {
				// We cannot comply. Let the higher layer decide what
				// to do about that.
				break;
			}
		}

		return total;
	}

	virtual void readFully(void *data, ::size_t count) {
		::size_t total = 0;
		while (total < count) {
			::size_t rd = read((char *)data + total, count - total);
			if (rd == 0) {
				throw mcmc::IOException("readFully cannot comply");
			}
			total += rd;
		}
	}

	/**
	 * @return bytes; -1 if size cannot be determined
	 */
	virtual ::ssize_t getSize() const {
		return -1;
	}

	virtual bool hasMore() = 0;
	virtual void close() = 0;
};


class WriterInterface {
public:
	virtual ~WriterInterface() {
	}

	virtual void write(const void *data, size_t count) = 0;

	//** default implementation for classes that don't support scatter/gather
	virtual void writev(const struct iovec *iov, int iovlen) {
		for (int i = 0; i < iovlen; i++) {
			write(iov[i].iov_base, iov[i].iov_len);
		}
	}

	virtual void close() = 0;

	//** support for buffered writer
	virtual void flush() {
	}

	//** flush if current capacity is less than <code>capacity</code>
	virtual void ensure(::size_t capacity __attribute__ ((unused))) {
	}

	inline void lock() {
		monitor.lock();
	}

	inline void unlock() {
		monitor.unlock();
	}

protected:
	boost::mutex monitor;
};

}		// namespace mr

#endif /* MR_IO_H_ */
