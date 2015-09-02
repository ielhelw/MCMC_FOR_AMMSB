#include "mcmc/options.h"

namespace mcmc {

::size_t parse_size_t(const std::string &argString) {
  ::size_t n = 0;
  const char *arg = argString.c_str();
  int base = 10;

  if (strncmp(arg, "0x", strlen("0x")) == 0 ||
      strncmp(arg, "0X", strlen("0X")) == 0) {
    base = 16;
    arg += 2;
  } else if (arg[0] == '0') {
    base = 8;
    arg++;
  }

  while (*arg != '\0') {
    if (base == 16 && isxdigit(*arg)) {
      int a;
      if (*arg >= '0' && *arg <= '9') {
        a = *arg - '0';
      } else if (*arg >= 'a' && *arg <= 'f') {
        a = *arg - 'a' + 10;
      } else {
        assert(*arg >= 'A' && *arg <= 'F');
        a = *arg - 'A' + 10;
      }
      if ((std::numeric_limits< ::size_t>::max() - a) / base < n) {
        throw mcmc::NumberFormatException("Overflow in parse_size_t");
      }
      n = a + n * base;
    } else if (base <= 10 && isdigit(*arg)) {
      int a = *arg - '0';
      if ((std::numeric_limits< ::size_t>::max() - a) / base < n) {
        throw mcmc::NumberFormatException("Overflow in parse_size_t");
      }
      n = a + n * base;
    } else if (strcasecmp(arg, "g") == 0 || strcasecmp(arg, "gb") == 0) {
      if ((std::numeric_limits< ::size_t>::max() >> 30) < n) {
        throw mcmc::NumberFormatException("Overflow in parse_size_t");
      }
      n *= 1ULL << 30;
      break;
    } else if (strcasecmp(arg, "m") == 0 || strcasecmp(arg, "mb") == 0) {
      if ((std::numeric_limits< ::size_t>::max() >> 20) < n) {
        throw mcmc::NumberFormatException("Overflow in parse_size_t");
      }
      n *= 1ULL << 20;
      break;
    } else if (strcasecmp(arg, "k") == 0 || strcasecmp(arg, "kb") == 0) {
      if ((std::numeric_limits< ::size_t>::max() >> 10) < n) {
        throw mcmc::NumberFormatException("Overflow in parse_size_t");
      }
      n *= 1ULL << 10;
      break;
    } else {
      std::ostringstream s;
      s << "Unknown characters in number: '" << *arg << "'";
      throw mcmc::NumberFormatException(s.str());
    }
    arg++;
  }

  return n;
}

}  // namespace mcmc
