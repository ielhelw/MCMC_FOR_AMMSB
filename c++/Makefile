PROJECT = $(shell cd .; pwd)

SUBDIRS += src
SUBDIRS += apps
SUBDIRS += test

include $(PROJECT)/include.mk

$(PROJECT)/config.mk: $(PROJECT)/config.mk.in
	cp $< $@
