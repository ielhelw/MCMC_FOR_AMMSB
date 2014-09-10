PROJECT = $(shell cd .; pwd)

SUBDIRS += src
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	if [ -e $@.orig ]; then diff -u $@.in $@.orig > $@.patch || echo "Nothing"; fi
	cp $< $@.orig
	cp $< $@
	if [ -e $@.patch ]; then patch -p0 < $@.patch && rm $@.patch; fi
