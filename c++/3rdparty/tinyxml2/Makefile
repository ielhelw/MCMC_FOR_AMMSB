SUBDIRS	+= src

.PHONY: default clean

default clean:
	for d in $(SUBDIRS); do \
		cd $$d; $(MAKE) $(MFLAGS) $@; \
	done
