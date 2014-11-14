PROJECT = $(shell cd .; pwd)

SUBDIRS += 3rdparty
SUBDIRS += src
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

apps test: src
src: 3rdparty

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	-cp $< $@
	-if [ -e $@.patch ]; then patch < $@.patch && diff -u $@.in $@ > $@.patch; fi
