/*
 * exception.h
 *
 *  Created on: Mar 6, 2012
 *      Author: ielhelw
 */

#ifndef MCMC_EXCEPTION_H_
#define MCMC_EXCEPTION_H_

#include <errno.h>
#include <string.h>

#include <exception>
#include <string>
#include <sstream>


namespace mcmc {

class MCMCException : public std::exception {
public:
	MCMCException(const std::string &reason) throw()
			: reason(reason) {
	}

	virtual ~MCMCException() throw() {
	}

	virtual const char *what() const throw() {
		return reason.c_str();
	}

protected:
	MCMCException() throw()
			: reason("<apparently inherited>") {
	}

protected:
	std::string reason;
};


#if 0
class OutOfMemoryException : public MCMCException {
public:
	OutOfMemoryException() {
	}

	OutOfMemoryException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class InvalidArgumentException : public MCMCException {
public:
	InvalidArgumentException() {
	}

	InvalidArgumentException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class BufferSizeException : public MCMCException {
public:
	BufferSizeException() {
	}

	BufferSizeException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class CorruptedStateException : public MCMCException {
public:
	CorruptedStateException() {
	}

	CorruptedStateException(const std::string &reason)
			: MCMCException(reason) {
	}
};
#endif


class UnimplementedException : public MCMCException {
public:
	UnimplementedException() {
	}

	UnimplementedException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class OutOfRangeException : public MCMCException {
public:
	OutOfRangeException() {
	}

	OutOfRangeException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class MalformattedException : public MCMCException {
public:
	MalformattedException() {
	}

	MalformattedException(const std::string &reason)
			: MCMCException(reason) {
	}
};


class NumberFormatException : public MCMCException {
public:
	NumberFormatException() {
	}

	NumberFormatException(const std::string &reason)
			: MCMCException(reason) {
	}
};

#if 0

class SystemException : public MCMCException {
public:
	SystemException() {
	}

	SystemException(const std::string &reason)
			: MCMCException(reason + ": " + std::string(strerror(errno))) {
	}
};
#endif

class IOException : public MCMCException {
public:
	IOException() {
	}

	IOException(const std::string &reason) : MCMCException(reason) {
	}
};

class FileException : public IOException {
public:
	FileException() {
	}

	FileException(const std::string &reason)
			: IOException(reason + ": " + std::string(strerror(errno))) {
	}
};

}


#endif /* MR_EXCEPTION_H_ */

