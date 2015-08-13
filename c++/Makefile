PROJECT = $(shell cd .; pwd)

GOOGLE_SPARSEHASH = 3rdparty/sparsehash-2.0.2/

SUBDIRS += 3rdparty
SUBDIRS += src
SUBDIRS += lib
SUBDIRS += apps
SUBDIRS += test

PRECIOUSDIRS =
PRECIOUSDIRS += 3rdparty/daslib
PRECIOUSDIRS += 3rdparty/tinyxml
PRECIOUSDIRS += $(GOOGLE_SPARSEHASH)

include $(PROJECT)/include.mk

.PHONY: src lib apps test

apps test: lib
src: 3rdparty
lib: src

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in $(GOOGLE_SPARSEHASH)/Makefile
	-if test -e $@ -a \! -e $@.patch ; then diff -u $@.in $@ > $@.patch; fi
	-cp $< $@
	-if test -e $@.patch ; then patch < $@.patch && diff -u $@.in $@ > $@.patch; fi

INSTALL = $(shell cd $(PROJECT); pwd)

$(GOOGLE_SPARSEHASH)/Makefile: $(GOOGLE_SPARSEHASH)/Makefile.in
	cd $(GOOGLE_SPARSEHASH); \
		touch Makefile.in aclocal.m4 config.h.in configure src/Makefile.in; \
		./configure --prefix=$(INSTALL) && $(MAKE) $(MFLAGS) && $(MAKE) $(MFLAGS) install
