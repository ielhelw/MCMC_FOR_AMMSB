#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

namespace mcmc {

class Random {
public:
	int randint(int from, int upto) {
		return -1;
	}

	template <class List>
	List *sample(const List &list, ::size_t count) {
		List *result = new List();
		for (::size_t i = 0; i < count; i++) {
			int r = randint(0, list->size());
			if (result->find(r) == result->end()) {
				result->insert(r);
			} else {
				i--;
			}
		}

		return result;
	}


	template <class List>
	List *sample(const List *list, ::size_t count) {
		return sample(*list, count);
	}
};

extern Random random;

}	// namespace mcmc

#endif	// ndef MCMC_RANDOM_H__
