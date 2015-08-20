.PHONY:	default cc python clean

default:	cc python

cc:
	$(MAKE) $(MFLAGS) -C c++

python:
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

