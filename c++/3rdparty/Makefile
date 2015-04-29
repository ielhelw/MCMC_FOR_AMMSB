SOURCE_SUBDIRS	=
SOURCE_SUBDIRS	+= tinyxml2
SOURCE_SUBDIRS	+= daslib
SUBDIRS		=
SUBDIRS		+= vexcl

default: $(SUBDIRS) $(SOURCE_SUBDIRS)
	for d in $(SOURCE_SUBDIRS); do (cd $$d; $(MAKE) $(MFLAGS)); done

vexcl:
	if [ \! -d vexcl ] ; then git clone https://github.com/ddemidov/vexcl.git; fi

clean:
	@echo "make clean in 3rdparty: ignored"
