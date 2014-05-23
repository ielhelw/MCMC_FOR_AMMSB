

all:
	python setup.py install_lib --install-dir=.

clean:
	rm -rf build
	rm -f com/uva/sample_latent_vars.c
	rm -f com/uva/sample_latent_vars.so
