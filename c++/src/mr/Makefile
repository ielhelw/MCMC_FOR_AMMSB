PROJECT = ../..

include $(PROJECT)/config.mk

ifneq (0, $(CONFIG_NETWORKING))
SUBDIRS	+= net
endif

CXX_SOURCES =
CXX_SOURCES += rdtsc_timer.cc

TARGET_LIBS = libmcmc

include $(PROJECT)/include.mk
