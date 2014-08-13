PROJECT		:= $(shell cd ..; pwd)

LDSO		:= gcc -shared -o

LIB		:= $(PROJECT)/lib/libtinyxml2.a
LIBSO		:= $(LIB:%.a=%.so)

CPPFLAGS	+= -I$(PROJECT)/include
CXXFLAGS	+= -DNDEBUG -O3 -finline-functions
CXXFLAGS	+= -g -Wall -fPIC

CPPFILES	:= $(wildcard *.cpp)
OBJ		:= $(CPPFILES:%.cpp=%.o)

default:	lib

.PHONY:		lib
lib:		$(LIB)

MAJOR		:= 0
MINOR		:= 98

$(LIB):		$(OBJ)
		-mkdir -p $(PROJECT)/lib
		$(AR) $(ARFLAGS) $(LIB) $(OBJ)
ifdef RANLIB
		$(RANLIB) $(LIB)
endif
ifdef LDSO
		-mkdir -p $(PROJECT)/lib/$(ARCH)/so
		cd $(PROJECT)/lib/$(ARCH)/so; \
			$(RM) *.o; \
			$(AR) x $(LIB); \
			$(LDSO) $(LIBSO).$(MAJOR) -Wl,-soname,$(LIBSO).$(MAJOR) *.o
		-rm -f $(LIBSO)
		-ln -s $(LIBSO).$(MAJOR) $(LIBSO)
endif

.PHONY: clean
clean:
	-rm -f $(OBJ) $(LIB) $(LIBSO).$(MAJOR) $(LIBSO)
