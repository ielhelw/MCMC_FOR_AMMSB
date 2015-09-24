.PHONY:	default python clean

default: python

python:
	@echo "On DAS, don't forget to set your PYTHONPATH to a recent Cython code drop"
	@echo "Don't forget to set your LD_LIBRARY_PATH to include $$MCMC/<build>/src"
	python setup.py install_lib --install-dir=.

clean:
	rm -rf build
	find com -name '*.pyc' -exec rm {} \;
	rm -f com/uva/sample_latent_vars.c
	rm -f com/uva/sample_latent_vars.so
	rm -f com/uva/estimate_phi.c
	rm -f com/uva/estimate_phi.so
	rm -f com/uva/custom_random/custom_random.so
	rm -f com/uva/custom_random/custom_random.cpp

