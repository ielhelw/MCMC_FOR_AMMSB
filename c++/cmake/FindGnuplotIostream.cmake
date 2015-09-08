# follows GNUPLOTIO_ROOT

#defines:
#- GNUPLOTIO_INCLUDE_DIRS

FIND_PACKAGE(PackageHandleStandardArgs)

if (DEFINED ENV{GNUPLOTIO_ROOT})
	set(GNUPLOTIO_ROOT "$ENV{GNUPLOTIO_ROOT}")
endif()

find_path(GNUPLOTIO_INCLUDE_DIRS gnuplot-iostream.h
  PATHS "${GNUPLOTIO_ROOT}"
  PATH_SUFFIXES include
)

find_package_handle_standard_args(GNUPLOTIO DEFAULT_MSG
    GNUPLOTIO_INCLUDE_DIRS)
