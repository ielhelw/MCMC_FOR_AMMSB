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

#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include <assert.h>

// #include "mrg_os.h"
#include "mrg_util.h"

#include "mrg_errno.h"
#include "mrg_util_error.h"


#define HAVE_LOG_FILE   0

int     mrg_verbose = 0;

#if HAVE_LOG_FILE
FILE *mrg_log_file;
#endif


void
mrg_error_pre(const char *file, int line)
{
    fprintf(stderr, "           [MRG] %s:%d *** Fatal error: ", file, line);
}


int
mrg_error_print(const char *fmt, ...)
{
    int         r;
    va_list     ap;

    va_start(ap, fmt);
    r = vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, ": %s\n", mrg_error_describe());
    if (errno != 0) {
        fprintf(stderr, "; OS errno = %d %s\n", errno, strerror(errno));
    }

#if 0
    exit(33);
#endif

    return r;
}


int
mrg_warning(const char *fmt, ...)
{
    int         r;
    va_list     ap;

    fprintf(stderr, "             [MRG] *** Warning: ");
    va_start(ap, fmt);
    r = vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");

    return r;
}


typedef struct MRG_ERROR_DESCR {
    const char *file;
    int         line;
    int         num;
    const char *msg;
} mrg_error_descr_t, *mrg_error_descr_p;


static mrg_error_descr_t        mrg_error_descr;


static mrg_error_descr_p
mrg_get_errno(void)
{
    mrg_error_descr_p error = &mrg_error_descr;

    return error;
}


int
mrg_error_set_annotated(const char *file, int line, int err, const char *msg)
{
    mrg_error_descr_p error = mrg_get_errno();

    error->file = file;
    error->line = line;
    error->num = err;
    error->msg = msg;

    return 0;
}


int
mrg_error_get(void)
{
    mrg_error_descr_p error = mrg_get_errno();

    return error->num;
}


const char *
mrg_error_describe(void)
{
#define ERROR_LENGTH    1024
    static char error_buf[ERROR_LENGTH];
    mrg_error_descr_p error = mrg_get_errno();

    memset(error_buf, '\0', sizeof error_buf);

    snprintf(error_buf, ERROR_LENGTH,
            "%s:%d [%d] %s",
            error->file, error->line, error->num, error->msg);

    return error_buf;
}


int
mrg_error_reset(void)
{
    mrg_error_descr_p error = mrg_get_errno();

    error->num = MRG_ERROR_OK;
    error->msg = "No error";
    
    errno = 0;

    return 0;
}


const char *
mrg_error_string(void)
{
    return mrg_errno_string(mrg_get_errno()->num);
}
