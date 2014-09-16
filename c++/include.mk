MAJOR	= 0
MINOR	= 23

PROJECT_HOME = $(shell cd $(PROJECT); pwd)

include $(PROJECT_HOME)/config.mk

LD = $(CXX)
LDSHARED = $(CXX)

CXXFLAGS += -std=c++0x
CXXFLAGS += -fPIC
ifeq (1, $(CONFIG_OPTIMIZE))
	CXXFLAGS += -g3 -O2 -finline-functions
	CXXFLAGS += -DNDEBUG
else
	CXXFLAGS += -g3
	CXXFLAGS += -fno-inline-functions
endif
ifeq (1, $(CONFIG_PROFILE))
	CXXFLAGS += -pg
	CXXFLAGS := $(filter-out -finline-functions, $(CXXFLAGS))
	CXXFLAGS += -fno-inline-functions
	LDFLAGS += -pg
	CONFIG_STATIC_LIB	= 1
	export GMON_OUT_PREFIX=gmon.out	# document: will save to gmon.out.<PID>
endif
CXXFLAGS += -Wall
ifneq (icpc, $(CXX))
CXXFLAGS += -Werror
else
CXXFLAGS += -no-gcc
endif
# may need to do this without -Werror:
# CXXFLAGS += -Wconversion
CXXFLAGS += -Wno-comment
CXXFLAGS += -Wunused
CXXFLAGS += -Wextra
# CXXFLAGS += -pedantic
# CXXFLAGS += -fmessage-length=0
CXXFLAGS += -Wno-unused-parameter
CXXFLAGS += -Wno-ignored-qualifiers
CXXFLAGS += -I$(PROJECT_HOME)/include
CXXFLAGS += -I$(PROJECT_HOME)/3rdparty/tinyxml2/include
ifneq (, $(OPENCL_ROOT))
CXXFLAGS += -I$(OPENCL_ROOT)/include
CXXFLAGS += -DENABLE_OPENCL
endif
CXXFLAGS += -DPROJECT_HOME=$(PROJECT_HOME)
ifneq (, $(BOOST_INCLUDE))
CXXFLAGS += -I$(BOOST_INCLUDE)
BOOST_ROOT = $(dir $(BOOST_INCLUDE))
endif

LDFLAGS += -L$(PROJECT_HOME)/lib -l mcmc
LDFLAGS += -L$(PROJECT_HOME)/3rdparty/tinyxml2/lib -ltinyxml2
ifneq (, $(OPENCL_ROOT))
LDFLAGS += -L$(OPENCL_ROOT)/lib -L$(OPENCL_ROOT)/lib/x86_64 -lOpenCL
endif
ifdef USE_MUDFLAP
LIBS	+= -lmudflapth -rdynamic
CXXFLAGS += -fmudflap -fmudflapth -funwind-tables
endif

LD_FLAGS_LIB_SHARED += -shared
#-Wl,-soname,libplugin.so.0
ifdef USE_MUDFLAP
LD_FLAGS_LIB_SHARED += -lmudflapth -rdynamic
endif

AR_FLAGS	= rc

ifneq (, $(BOOST_ROOT))
LDFLAGS += -L$(BOOST_ROOT)/lib
endif
LIBS	+= -lboost_system-mt
LIBS	+= -lboost_thread-mt
LIBS	+= -lboost_filesystem-mt
LIBS	+= -lboost_program_options-mt

vpath lib%.so	$(LD_LIBRARY_PATH) $(subst -L,,$(LDFLAGS))
vpath lib%.a	$(LD_LIBRARY_PATH) $(subst -L,,$(LDFLAGS))

OBJDIR = obj
LIBDIR = $(PROJECT_HOME)/lib

CXX_OBJECTS += $(CXX_SOURCES:%.cc=$(OBJDIR)/%.o)
CXX_OBJECTS += $(CPP_SOURCES:%.cpp=$(OBJDIR)/%.o)
CLEANSUBDIRS = $(SUBDIRS:%=%.clean)
DEPENDS = $(CXX_OBJECTS:%.o=%.d) $(TARGETS:%=$(OBJDIR)/%.d)

.PHONY: all clean subdirs $(SUBDIRS) cleansubdirs $(CLEANSUBDIRS) depends
.PHONY: $(TARGET_LIBS)

all: $(CXX_OBJECTS) $(TARGET_LIBS) $(TARGETS) subdirs

lib: $(CXX_OBJECTS) $(TARGET_LIBS) $(TARGETS) $(filter-out test, $(filter-out apps, $(SUBDIRS)))

$(OBJDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(TARGETS): % : $(OBJDIR)/%.o $(CXX_OBJECTS) $(LIBS)
	$(LD) $< $(CXX_OBJECTS) $(LDFLAGS) $(LIBS) -o $@

TARGET_LIBS_SHARED	= $(TARGET_LIBS:%=$(LIBDIR)/%.so)
TARGET_LIBS_STATIC	= $(TARGET_LIBS:%=$(LIBDIR)/%.a)

ifeq (1, $(CONFIG_STATIC_LIB))
$(TARGET_LIBS):	$(TARGET_LIBS_STATIC)
# LDFLAGS	+= -static
LDFLAGS	+= -static-libgcc
else
$(TARGET_LIBS):	$(TARGET_LIBS_SHARED)
endif

$(TARGET_LIBS_STATIC): $(CXX_OBJECTS)
	@mkdir -p $(LIBDIR)
	$(AR) $(AR_FLAGS) $@ $^

$(TARGET_LIBS_SHARED): $(CXX_OBJECTS)
	@mkdir -p $(LIBDIR)
	$(LDSHARED) $(LD_FLAGS_LIB_SHARED) $^ -o $@.$(MAJOR).$(MINOR) -Wl,-soname,$@.$(MAJOR).$(MINOR)
	rm -f $@.$(MAJOR) $@
	ln -s $@.$(MAJOR).$(MINOR) $@.$(MAJOR)
	ln -s $@.$(MAJOR) $@

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

runtests:
ifeq (apps, $(findstring apps, $(SUBDIRS)))
	cd apps; $(MAKE) $(MFLAGS) RUNTESTS_FLAGS="$(RUNTESTS_FLAGS)" $@
else ifneq (, $(SUBDIRS))
	for d in $(SUBDIRS); do (cd $$d; $(MAKE) $(MFLAGS) RUNTESTS_FLAGS="$(RUNTESTS_FLAGS)" $@); done
else ifeq (runtests.bash, $(wildcard runtests.bash))
	./runtests.bash $(RUNTESTS_FLAGS)
endif

cleansubdirs: $(CLEANSUBDIRS)

$(CLEANSUBDIRS):
	$(MAKE) -C $(@:%.clean=%) clean

clean: cleansubdirs
	rm -rf $(TARGETS) $(TARGETS:%=%.o) $(TARGET_LIBS_SHARED) $(TARGET_LIBS_STATIC) $(CXX_OBJECTS) $(DEPENDS) $(OBJDIR)

.PHONY:	ALWAYS distclean

ALWAYS:

distclean: clean ALWAYS
	-rm -f config.cache Makefile.config
	-rm -f OpenCLInclude/generated/{emit.h,memcpy.h}

depends: $(DEPENDS)

build:	default

default: subdirs

-include $(DEPENDS)
