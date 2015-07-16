# follows GLOG_ROOT

#defines:
#- GLOG_INCLUDE_DIRS
#- GLOG_LIBRARIES

FIND_PACKAGE(PackageHandleStandardArgs)

if (DEFINED ENV{GLOG_ROOT})
	set(GLOG_ROOT "$ENV{GLOG_ROOT}")
endif()

find_path(GLOG_INCLUDE_DIRS glog/logging.h
  PATHS "${GLOG_ROOT}"
  PATH_SUFFIXES include
)

find_library(GLOG_LIBRARIES NAMES glog
  PATHS ${GLOG_ROOT}
  PATH_SUFFIXES lib libs
)

find_package_handle_standard_args(GLOG DEFAULT_MSG
    GLOG_INCLUDE_DIRS GLOG_LIBRARIES)
