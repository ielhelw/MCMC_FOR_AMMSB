PROJECT = $(shell cd .; pwd)

SUBDIRS += src
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	-diff -u $@.in $@ > $@.patch
	cp $< $@
	patch -p0 < $@.patch && rm $@.patch
