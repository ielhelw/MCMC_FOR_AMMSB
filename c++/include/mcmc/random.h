#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

namespace mcmc {

class Random {
public:
	int randint(int from, int upto);

	template <class List>
	List *sample(const List &list, ::size_t count);

	template <class List>
	List *sample(const List *list, ::size_t count) {
		return sample(*list, count);
	}
};

extern Random random;

}	// namespace mcmc

#endif	// ndef MCMC_RANDOM_H__
