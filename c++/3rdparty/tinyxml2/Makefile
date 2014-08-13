SUBDIRS	+= src

.PHONY: default

default clean:
	for d in $(SUBDIRS); do \
		cd $$d; $(MAKE) $(MFLAGS) $@; \
	done
