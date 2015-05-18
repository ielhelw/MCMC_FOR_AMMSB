/*
 * Copyright 2007, Rutger Hofman, VU Amsterdam
 *
 * This file is part of the RFID Guardian Project Software.
 *
 * The RFID Guardian Project Software is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, version 3 of the License.
 *
 * The RFID Guardian Project Software is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with The RFID Guardian Project Software.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef MRG_UTIL_H__
#define MRG_UTIL_H__

/**
 * @file mrg_util.h
 *
 * @brief Utility module. See also mrg_errno.h
 *
 * Does:
 *  - debug print control
 *  - bit string manipulation
 *  - some compilers think a <code>signed char *</code> is not parameter-
 *    compatible with a <code>char *</code>. For this we need mrg_int8_string
 *  - data streams: RFID streams have more info than can be encoded in a byte:
 *    SOF/EOF markers, partial bytes, collision bits.
 *
 * @author 2005, 2006, 2007 Rutger Hofman, VU Amsterdam
 */

#include <stddef.h>

#include "mrg_errno.h"

/**
 * @defgroup verbose Debug/Verbose implementation
 * @{
 */

/**
 * Macro to print an 'unimplemented' message on diag.
 */
#define MRG_UNIMPLEMENTED(func) \
    ( fprintf(stderr, "%s:%d MRG: unimplemented routine %s -- PLEASE IMPLEMENT\n", \
            __FILE__, __LINE__, #func), \
      mrg_error_set(MRG_ERROR_UNIMPLEMENTED, #func), \
      -1 )

#endif
