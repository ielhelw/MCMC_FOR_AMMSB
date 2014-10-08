PROJECT = $(shell cd .; pwd)

SUBDIRS += src
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	-if [ -e $@.orig ]; then diff -u $@.orig $@ > $@.patch || echo "Nothing"; fi
	-cp $< $@.orig
	-cp $< $@
	-if [ -e $@.patch ]; then patch < $@.patch && rm $@.patch; fi
