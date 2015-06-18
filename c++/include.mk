MAJOR	= 0
MINOR	= 23

PROJECT_HOME = $(shell cd $(PROJECT); pwd)

include $(PROJECT_HOME)/config.mk

ifeq (1, $(CONFIG_OPENMP))
CANYFLAGS	+= -fopenmp
LDFLAGS		+= -fopenmp
else
CANYFLAGS	+= -Wno-unknown-pragmas
endif

LD = $(CXX)

CXXFLAGS += -std=c++0x
CANYFLAGS += -fPIC
CFLAGS += -fPIC
ifeq (1, $(CONFIG_OPTIMIZE))
	CANYFLAGS += -g3 -O3 # -finline-functions
	CPPFLAGS += -DNDEBUG
else
	CANYFLAGS += -g3 -O0
	CANYFLAGS += -fno-inline-functions
endif
ifeq (1, $(CONFIG_PROFILE))
	CANYFLAGS += -pg
	CANYFLAGS := $(filter-out -finline-functions, $(CANYFLAGS))
	CANYFLAGS += -fno-inline -fno-inline-functions
	LDFLAGS += -pg
	CONFIG_STATIC_LIB	= 1
	export GMON_OUT_PREFIX=gmon.out	# document: will save to gmon.out.<PID>
endif

CANYFLAGS += -Wall
ifneq (icpc, $(CXX))
CANYFLAGS += -Werror
CXXFLAGS += -Wno-literal-suffix
else
CANYFLAGS += -no-gcc
endif
# may need to do this without -Werror:
# CXXFLAGS += -Wconversion
CANYFLAGS += -Wunused
CANYFLAGS += -Wextra
# CANYFLAGS += -pedantic
# CANYFLAGS += -fmessage-length=0
CANYFLAGS += -Wno-unused-parameter

CPPFLAGS += -I$(PROJECT_HOME)/include
CPPFLAGS += -I$(PROJECT_HOME)/3rdparty/tinyxml2/include
ifneq (, $(OPENCL_ROOT))
CPPFLAGS += -I$(OPENCL_ROOT)/include
CPPFLAGS += -DENABLE_OPENCL
endif
CPPFLAGS += -DPROJECT_HOME=$(PROJECT_HOME)
ifneq (, $(BOOST_INCLUDE))
CPPFLAGS += -I$(BOOST_INCLUDE)
BOOST_ROOT = $(dir $(BOOST_INCLUDE))
endif
ifeq (1, $(CONFIG_OPENMP))
CPPFLAGS += -DENABLE_OPENMP
endif
ifeq (1, $(CONFIG_DISTRIBUTED))
CPPFLAGS += -DENABLE_DISTRIBUTED
# OpenMPI requires this
endif
ifeq (0, $(CONFIG_NETWORKING))
CPPFLAGS += -DDISABLE_NETWORKING
else
CPPFLAGS += -I$(PROJECT_HOME)/3rdparty/daslib/include
LIB_LDFLAGS += -L$(PROJECT_HOME)/3rdparty/daslib/lib/$(shell uname -m)_$(shell uname -s)
LIB_LDLIBS += -ldas
endif

CFLAGS += -std=gnu99

ifneq (, $(OPENCL_ROOT))
VEXCL		= $(PROJECT)/3rdparty/vexcl
CXXFLAGS	+= -I$(VEXCL)
endif

LDFLAGS += -L$(PROJECT_HOME)/lib
LDLIBS += -lmcmc
LDFLAGS	+= -Wl,-rpath,$(PROJECT_HOME)/lib
LIB_LDFLAGS += -L$(PROJECT_HOME)/3rdparty/tinyxml2/lib
LIB_LDLIBS += -ltinyxml2
LIB_LDFLAGS	+= -Wl,-rpath,$(PROJECT_HOME)/3rdparty/tinyxml2/lib
# export LD_RUN_PATH := $(PROJECT_HOME)/lib
ifneq (, $(OPENCL_ROOT))
LIB_LDFLAGS += -L$(OPENCL_ROOT)/lib -L$(OPENCL_ROOT)/lib/x86_64
LIB_LDLIBS += -lOpenCL
endif
ifdef USE_MUDFLAP
LDLIBS	+= -lmudflapth -rdynamic
CANYFLAGS += -fmudflap -fmudflapth -funwind-tables
endif

LD_FLAGS_LIB_SHARED += -shared
#-Wl,-soname,libplugin.so.0
ifdef USE_MUDFLAP
LD_FLAGS_LIB_SHARED += -lmudflapth -rdynamic
endif

ifdef USE_ADDRESS_SANTIZER
CANYFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address
endif

AR_FLAGS	= rc

ifneq (, $(BOOST_ROOT))
LIB_LDFLAGS += -L$(BOOST_ROOT)/lib
endif
LIB_LDLIBS	+= -lboost_system$(BOOST_SUFFIX)
LIB_LDLIBS	+= -lboost_filesystem$(BOOST_SUFFIX)
LIB_LDLIBS	+= -lboost_program_options$(BOOST_SUFFIX)
ifeq (1, $(CONFIG_RDMA))
ifneq (1, $(CONFIG_NETWORKING))
LIB_LDLIBS	+= -lboost_thread$(BOOST_SUFFIX)
endif
endif

ifneq (, $(CONFIG_RAMCLOUD_ROOT))
CPPFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/src
CPPFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/obj.master
CPPFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/gtest/include
CPPFLAGS += -DENABLE_RAMCLOUD
LIB_LDFLAGS += -L$(CONFIG_RAMCLOUD_ROOT)/obj.master
LIB_LDFLAGS += -Wl,-rpath,$(CONFIG_RAMCLOUD_ROOT)/obj.master
LIB_LDLIBS  += -lramcloud
# export LD_RUN_PATH := $(LD_RUN_PATH):$(CONFIG_RAMCLOUD_ROOT)/obj.master
endif
ifeq (1, $(CONFIG_RDMA))
CPPFLAGS += -DENABLE_RDMA
LIB_LDLIBS	+= -libverbs
endif

vpath lib%.so	$(LD_LIBRARY_PATH) $(subst -L,,$(LIB_LDFLAGS)) $(subst -L,,$(LDFLAGS))
vpath lib%.a	$(LD_LIBRARY_PATH) $(subst -L,,$(LIB_LDFLAGS)) $(subst -L,,$(LDFLAGS))

OBJDIR = $(PROJECT_HOME)/obj
LIBDIR = $(PROJECT_HOME)/lib

CXX_OBJECTS += $(CXX_SOURCES:%.cc=$(OBJDIR)/%.o)
CXX_OBJECTS += $(CPP_SOURCES:%.cpp=$(OBJDIR)/%.o)
CXX_OBJECTS += $(C_SOURCES:%.c=$(OBJDIR)/%.o)
CLEANSUBDIRS = $(SUBDIRS:%=%.clean)
DEPENDS = $(CXX_OBJECTS:%.o=%.d) $(TARGETS:%=$(OBJDIR)/%.d)

.PHONY: all clean subdirs $(SUBDIRS) cleansubdirs $(CLEANSUBDIRS) depends
.PHONY: $(TARGET_LIBS)

all: objects $(TARGET_LIBS) $(TARGETS) subdirs

lib: objects $(TARGET_LIBS) $(TARGETS) $(filter-out test, $(filter-out apps, $(SUBDIRS)))

.PHONY: objects
objects:	$(CXX_OBJECTS)

$(OBJDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CANYFLAGS) $(CPPFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CANYFLAGS) $(CPPFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CANYFLAGS) $(CPPFLAGS) -MMD -c $< -o $@

$(TARGETS): % : $(OBJDIR)/%.o $(CXX_OBJECTS) $(LDLIBS) $(LIB_LDLIBS)
	@echo LD_RUN_PATH $(LD_RUN_PATH)
	$(LD) $< $(CXX_OBJECTS) $(LDFLAGS) $(LIB_LDFLAGS) $(LDLIBS) $(LIB_LDLIBS) -o $@

TARGET_LIBS_SHARED	= $(TARGET_LIBS:%=$(LIBDIR)/%.so)
TARGET_LIBS_STATIC	= $(TARGET_LIBS:%=$(LIBDIR)/%.a)

$(TARGET_LIBS):	$(TARGET_LIBS_STATIC)
# LDFLAGS	+= -static
# LDFLAGS	+= -static-libgcc

$(TARGET_LIBS_STATIC): $(CXX_OBJECTS)
	# $(AR) $(AR_FLAGS) $@ $^

# $(TARGET_LIBS_SHARED): $(CXX_OBJECTS)
# 	$(LDSHARED) $(LD_FLAGS_LIB_SHARED) $^ -o $@.$(MAJOR).$(MINOR) -Wl,-soname,$@.$(MAJOR).$(MINOR)
# 	rm -f $@.$(MAJOR) $@
# 	ln -s $@.$(MAJOR).$(MINOR) $@.$(MAJOR)
# 	ln -s $@.$(MAJOR) $@

subdirs: $(SUBDIRS)

$(SUBDIRS):
	# $(MAKE) $(MFLAGS) -C $@
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
