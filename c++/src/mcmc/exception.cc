#include "mcmc/exception.h"

namespace mcmc {

MCMCException::MCMCException(const std::string &reason) throw()
    : reason(reason) {}

MCMCException::~MCMCException() throw() {}

const char *MCMCException::what() const throw() { return reason.c_str(); }

MCMCException::MCMCException() throw() : reason("<apparently inherited>") {}

InvalidArgumentException::InvalidArgumentException() {}

InvalidArgumentException::InvalidArgumentException(const std::string &reason)
    : MCMCException(reason) {}

BufferSizeException::BufferSizeException() {}

BufferSizeException::BufferSizeException(const std::string &reason)
    : MCMCException(reason) {}

UnimplementedException::UnimplementedException() {}

UnimplementedException::UnimplementedException(const std::string &reason)
    : MCMCException(reason) {}

OutOfRangeException::OutOfRangeException() {}

OutOfRangeException::OutOfRangeException(const std::string &reason)
    : MCMCException(reason) {}

MalformattedException::MalformattedException() {}

MalformattedException::MalformattedException(const std::string &reason)
    : MCMCException(reason) {}

NumberFormatException::NumberFormatException() {}

NumberFormatException::NumberFormatException(const std::string &reason)
    : MCMCException(reason) {}

IOException::IOException() {}

IOException::IOException(const std::string &reason) : MCMCException(reason) {}

FileException::FileException() {}

FileException::FileException(const std::string &reason)
    : IOException(reason + ": " + std::string(strerror(errno))) {}

}  // namespace mcmc
