#ifndef MCMC_FIXED_SIZE_SET__
#define MCMC_FIXED_SIZE_SET__

#include <vector>

namespace mcmc {

class FixedSizeSet {
 public:
  typedef int32_t   key_type;
  typedef key_type  value_type;
  typedef int32_t   index_type;

  class iterator_type {
    public:
     typedef FixedSizeSet::key_type key_type;
     typedef FixedSizeSet::index_type index_type;

     iterator_type(const FixedSizeSet& set, index_type x)
        : set_(set), x_(x) {
     }

     iterator_type& operator++() {
       do {
         ++x_;
       } while (x_ < static_cast<index_type>(set_.hash_.size()) &&
                  set_.hash_[x_] == set_.SENTINEL);

       return *this;
     }

     bool operator==(const iterator_type& other) {
       return x_ == other.x_;
     }

     bool operator!=(const iterator_type& other) {
       return x_ != other.x_;
     }

     key_type operator*() {
       return set_.hash_[x_];
     }

    private:
     const FixedSizeSet& set_;
     index_type x_;
  };

  FixedSizeSet(::size_t n_elements) : size_(0), end_(*this, 2 * twopow(n_elements)) {
    hash_.resize(2 * twopow(n_elements), SENTINEL);
  }

  void insert(key_type x) {
    ++size_;
    index_type i = hash(x) % hash_.size();
    if (hash_[i] == SENTINEL) {
      hash_[i] = x;
    } else {
      index_type i2 = hash2(x);
      while (true) {
        i = (i + i2) % hash_.size();
        if (hash_[i] == SENTINEL) {
          hash_[i] = x;
          break;
        }
      }
    }
  }

  iterator_type find(key_type x) const {
    index_type i = hash(x) % hash_.size();
    if (hash_[i] == SENTINEL) {
      return end();
    }
    if (hash_[i] == x) {
      return iterator_type(*this, i);
    }

    index_type i2 = hash2(x);
    while (true) {
      i = (i + i2) % hash_.size();
      if (hash_[i] == SENTINEL) {
        return end();
      }
      if (hash_[i] == x) {
        return iterator_type(*this, i);
      }
    }
  }

  iterator_type begin() const {
    index_type x = 0;
    while (x < static_cast<key_type>(hash_.size()) && hash_[x] == SENTINEL) {
      x++;
    }
    return iterator_type(*this, x);
  }

  const iterator_type& end() const {
    return end_;
  }

  ::size_t size() const {
    return size_;
  }

 private:
  const key_type SENTINEL = -1;

  static index_type hash(const key_type& x) {
    return x;
  }

  static index_type hash2(const key_type& x) {
    return x % 7 + 1;
  }

  static index_type twopow(index_type v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
  }

  std::vector<key_type> hash_;
  std::size_t size_;
  iterator_type end_;
};

}

#endif  // ndef MCMC_FIXED_SIZE_SET__
