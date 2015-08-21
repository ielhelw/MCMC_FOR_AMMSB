#ifndef MCMC_EXCEPTION_H_
#define MCMC_EXCEPTION_H_

#include <errno.h>
#include <string.h>

#include <exception>
#include <string>
#include <sstream>

#include "mcmc/config.h"

namespace mcmc {

class MCMCException : public std::exception {
 public:
  MCMCException(const std::string &reason) throw();

  virtual ~MCMCException() throw();

  virtual const char *what() const throw();

 protected:
  MCMCException() throw();

 protected:
  std::string reason;
};

class InvalidArgumentException : public MCMCException {
 public:
  InvalidArgumentException();

  InvalidArgumentException(const std::string &reason);
};

class BufferSizeException : public MCMCException {
 public:
  BufferSizeException();

  BufferSizeException(const std::string &reason);
};

class UnimplementedException : public MCMCException {
 public:
  UnimplementedException();

  UnimplementedException(const std::string &reason);
};

class OutOfRangeException : public MCMCException {
 public:
  OutOfRangeException();

  OutOfRangeException(const std::string &reason);
};

class MalformattedException : public MCMCException {
 public:
  MalformattedException();

  MalformattedException(const std::string &reason);
};

class NumberFormatException : public MCMCException {
 public:
  NumberFormatException();

  NumberFormatException(const std::string &reason);
};

class IOException : public MCMCException {
 public:
  IOException();

  IOException(const std::string &reason);
};

class FileException : public IOException {
 public:
  FileException();

  FileException(const std::string &reason);
};

}  // namespace mcmc

#endif  // MCMC_EXCEPTION_H_
