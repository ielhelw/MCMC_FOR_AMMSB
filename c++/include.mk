MAJOR	= 0
MINOR	= 23

PROJECT_HOME = $(shell cd $(PROJECT); pwd)

include $(PROJECT_HOME)/config.mk

ifeq (1, $(CONFIG_OPENMP))
CXXFLAGS		+= -fopenmp
LDFLAGS			+= -fopenmp
else
CXXFLAGS		+= -Wno-unknown-pragmas
endif

LD = $(CXX)

CXXFLAGS += -std=c++0x
CXXFLAGS += -fPIC
ifeq (1, $(CONFIG_OPTIMIZE))
	CXXFLAGS += -g3 -O3 # -finline-functions
	CXXFLAGS += -DNDEBUG
else
	CXXFLAGS += -g3 -O0
	CXXFLAGS += -fno-inline-functions
endif
ifeq (1, $(CONFIG_PROFILE))
	CXXFLAGS += -pg
	CXXFLAGS := $(filter-out -finline-functions, $(CXXFLAGS))
	CXXFLAGS += -fno-inline -fno-inline-functions
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
CXXFLAGS += -Wunused
CXXFLAGS += -Wextra
# CXXFLAGS += -pedantic
# CXXFLAGS += -fmessage-length=0
CXXFLAGS += -Wno-unused-parameter

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
ifeq (1, $(CONFIG_DISTRIBUTED))
CXXFLAGS += -DENABLE_DISTRIBUTED
# OpenMPI requires this
CXXFLAGS += -Wno-literal-suffix
endif

ifneq (, $(OPENCL_ROOT))
VEXCL		= $(PROJECT)/3rdparty/vexcl
CXXFLAGS	+= -I$(VEXCL)
endif

LDFLAGS += -L$(PROJECT_HOME)/lib -lmcmc
LDFLAGS	+= -Wl,-rpath,$(PROJECT_HOME)/lib
LDFLAGS += -L$(PROJECT_HOME)/3rdparty/tinyxml2/lib -ltinyxml2
LDFLAGS	+= -Wl,-rpath,$(PROJECT_HOME)/3rdparty/tinyxml2/lib
# export LD_RUN_PATH := $(PROJECT_HOME)/lib
ifneq (, $(OPENCL_ROOT))
LDFLAGS += -L$(OPENCL_ROOT)/lib -L$(OPENCL_ROOT)/lib/x86_64 -lOpenCL
endif
ifdef USE_MUDFLAP
LDLIBS	+= -lmudflapth -rdynamic
CXXFLAGS += -fmudflap -fmudflapth -funwind-tables
endif

LD_FLAGS_LIB_SHARED += -shared
#-Wl,-soname,libplugin.so.0
ifdef USE_MUDFLAP
LD_FLAGS_LIB_SHARED += -lmudflapth -rdynamic
endif

ifdef USE_ADDRESS_SANTIZER
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address
endif

AR_FLAGS	= rc

ifneq (, $(BOOST_ROOT))
LDFLAGS += -L$(BOOST_ROOT)/lib
endif
LDLIBS	+= -lboost_system$(BOOST_SUFFIX)
LDLIBS	+= -lboost_filesystem$(BOOST_SUFFIX)
LDLIBS	+= -lboost_program_options$(BOOST_SUFFIX)

ifneq (, $(CONFIG_RAMCLOUD_ROOT))
CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/src
CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/obj.master
CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/gtest/include
CXXFLAGS += -DENABLE_RAMCLOUD
LDFLAGS += -L$(CONFIG_RAMCLOUD_ROOT)/obj.master
LDFLAGS	+= -Wl,-rpath,$(CONFIG_RAMCLOUD_ROOT)/obj.master
LDLIBS  += -lramcloud
# export LD_RUN_PATH := $(LD_RUN_PATH):$(CONFIG_RAMCLOUD_ROOT)/obj.master
endif
ifeq (1, $(CONFIG_RDMA))
CXXFLAGS += -DENABLE_RDMA
LDLIBS	+= -libverbs
endif

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

$(TARGETS): % : $(OBJDIR)/%.o $(CXX_OBJECTS) $(LDLIBS)
	@echo LD_RUN_PATH $(LD_RUN_PATH)
	$(LD) $< $(CXX_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

TARGET_LIBS_SHARED	= $(TARGET_LIBS:%=$(LIBDIR)/%.so)
TARGET_LIBS_STATIC	= $(TARGET_LIBS:%=$(LIBDIR)/%.a)

$(TARGET_LIBS):	$(TARGET_LIBS_STATIC)
# LDFLAGS	+= -static
# LDFLAGS	+= -static-libgcc

$(TARGET_LIBS_STATIC): $(CXX_OBJECTS)
	$(AR) $(AR_FLAGS) $@ $^

$(TARGET_LIBS_SHARED): $(CXX_OBJECTS)
	$(LDSHARED) $(LD_FLAGS_LIB_SHARED) $^ -o $@.$(MAJOR).$(MINOR) -Wl,-soname,$@.$(MAJOR).$(MINOR)
	rm -f $@.$(MAJOR) $@
	ln -s $@.$(MAJOR).$(MINOR) $@.$(MAJOR)
	ln -s $@.$(MAJOR) $@

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) $(MFLAGS) -C $@

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
	$(MAKE) $(MFLAGS) -C $(@:%.clean=%) clean

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
