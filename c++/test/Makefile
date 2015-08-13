PROJECT = $(shell cd ..; pwd)

SUBDIRS += preprocess
SUBDIRS += rand-normal
SUBDIRS += rand-distr
ifeq (1, $(CONFIG_RDMA))
SUBDIRS += rdma
endif
SUBDIRS += d-kv-store

include $(PROJECT)/include.mk
