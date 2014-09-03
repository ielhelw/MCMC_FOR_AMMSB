

all:
	python setup.py install_lib --install-dir=.

clean:
	rm -rf build
	find com -name '*.pyc' -exec rm {} \;
	rm -f com/uva/sample_latent_vars.c
	rm -f com/uva/sample_latent_vars.so
	rm -f com/uva/sample_latent_vars.c
	rm -f com/uva/sample_latent_vars.so

