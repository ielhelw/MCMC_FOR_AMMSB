PROJECT	= $(shell cd ..; pwd)

OBJECT_DIR	= $(PROJECT)/obj

TARGET_STATIC	:= $(wildcard *.a)
# TARGET_SHARED	:= $(TARGET_STATIC:%.a=%.so)
TARGET_SHARED	:= libmcmc.so

.PHONY: default
default:	$(TARGET_SHARED)
# default:	$(TARGET_STATIC)

include $(PROJECT)/include.mk

# for clean
TARGET_LIBS_SHARED	:= $(TARGET_SHARED) $(TARGET_SHARED).$(MAJOR) $(TARGET_SHARED).$(MAJOR).$(MINOR)

.PHONY: ar_obj
ar_obj:
	-mkdir -p $(OBJECT_DIR)
	(cd $(OBJECT_DIR); $(AR) x ../$(TARGET_STATIC))

OBJECTS	= $(wildcard $(OBJECT_DIR)/*.o)

$(TARGET_SHARED):	$(OBJECTS) # ar_obj
	-rm -f $@.$(MAJOR).$(MINOR)
	$(LDSHARED) $(LD_FLAGS_LIB_SHARED) $(LIB_LDFLAGS) $(wildcard $(OBJECT_DIR)/*.o) -o $@.$(MAJOR).$(MINOR) -Wl,-soname,$@.$(MAJOR).$(MINOR) $(LIB_LDLIBS)
	# -rm -rf $(OBJECT_DIR)
	rm -f $@.$(MAJOR) $@
	ln -s $@.$(MAJOR).$(MINOR) $@.$(MAJOR)
	ln -s $@.$(MAJOR) $@


