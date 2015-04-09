PROJECT = ../../..

CXX_SOURCES =
CXX_SOURCES += DKVStoreRamCloud.cc

# CXXFLAGS += -I../../../include
CXXFLAGS += -fPIC

CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/src
CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/obj.master
CXXFLAGS += -I$(CONFIG_RAMCLOUD_ROOT)/gtest/include
CXXFLAGS += -Wno-unused-variable

TARGET_LIBS = libmcmc

include $(PROJECT)/include.mk
