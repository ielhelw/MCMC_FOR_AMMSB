SOURCE_SUBDIRS	=
SOURCE_SUBDIRS	+= daslib
SUBDIRS		=
SUBDIRS		+= vexcl

.PHONY: default
default: $(SUBDIRS) $(SOURCE_SUBDIRS)
	for d in $(SOURCE_SUBDIRS); do (cd $$d; $(MAKE) $(MFLAGS) $@); done

.PHONY: distclean
distclean: $(SUBDIRS) $(SOURCE_SUBDIRS)
	for d in $(SOURCE_SUBDIRS); do (cd $$d; $(MAKE) $(MFLAGS) clean); done

.PHONY: vexcl
vexcl:
	if [ \! -d vexcl ] ; then git clone https://github.com/ddemidov/vexcl.git; fi

.PHONY: clean
clean:
	@echo "make clean in 3rdparty: ignored"
