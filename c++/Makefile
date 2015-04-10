PROJECT = $(shell cd .; pwd)

SUBDIRS += 3rdparty
SUBDIRS += src
SUBDIRS += lib
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

.PHONY: src lib apps test

apps test: lib
src: 3rdparty
lib: src

lib:
	$(MAKE) $(MFLAGS) -C lib

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	-if test -e $@ -a \! -e $@.patch ; then diff -u $@.in $@ > $@.patch; fi
	-cp $< $@
	-if test -e $@.patch ; then patch < $@.patch && diff -u $@.in $@ > $@.patch; fi
