PROJECT = ../..

include $(PROJECT)/config.mk

SUBDIRS =
SUBDIRS += file
ifneq (, $(CONFIG_RAMCLOUD_ROOT))
SUBDIRS += ramcloud
endif
ifeq (1, $(CONFIG_RDMA))
SUBDIRS += rdma
endif

include $(PROJECT)/include.mk
