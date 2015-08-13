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

#ifndef MRG_ERRNO_H__
#define MRG_ERRNO_H__

/**
 * @file mrg_errno.h
 *
 * @brief Error handling.
 *
 * C has no exceptions (surprise!). Error handling is done by returning an
 * 'impossible' value (-1, NULL) and setting a global/glocal errno value
 * plus description string.
 * The layer that handles the error must also reset errno.
 *
 * @todo: errno must be glocal; not global.
 *
 * @author 2005, 2006, 2007 Rutger Hofman, VU Amsterdam
 */


/**
 * Possible errors
 */
typedef enum MRG_ERRNO {
    MRG_ERROR_OK,                       /**< no error */

    /** OS errors */
    MRG_ERROR_RESOURCE_EXHAUSTED,       /**< resource exhausted */
    MRG_ERROR_OUT_OF_MEMORY,            /**< out of memory */
    MRG_ERROR_SCHEDULER,                /**< scheduler error */
    MRG_ERROR_MUTEX,                    /**< mutex error */

    /** Interrupt errors */
    MRG_ERROR_INVALID_INTERRUPT,        /**< invalid interrupt */
    MRG_ERROR_NESTED_INTERRUPT,         /**< nested interrupt */

    /** Protocol errors */
    MRG_ERROR_NO_BUF,                   /**< no buffer */
    MRG_ERROR_NO_SUCH_COMMAND,          /**< undefined command */
    MRG_ERROR_NO_SUCH_PROTOCOL,         /**< undefined protocol */
    MRG_ERROR_PROTOCOL_UNSUPPORTED,     /**< unsupported protocol */
    MRG_ERROR_PROTOCOL,                 /**< protocol error */
    MRG_ERROR_MANCHESTER_CODING,        /**< manchester coding error */
    MRG_ERROR_AIR_STREAM_CORRUPTED,     /**< air stream corrupted */
    MRG_ERROR_INVALID_SYMBOL,           /**< invalid air stream symbol */

    /** Driver state */
    MRG_ERROR_INVALID_STATE,            /**< invalid state */
    MRG_ERROR_EARLY_ABORT,              /**< early abort */

    /** Device properties */
    MRG_ERROR_DEVICE_TOO_SLOW,          /**< device too slow */

    /** Generic frame errors */
    MRG_ERROR_CHECKSUM_FAILS,           /**< checksum error */
    MRG_ERROR_PARITY,                   /**< parity bit error */

    /** Port handling errors */
    MRG_ERROR_INTERRUPTED,              /**< call interrupted */
    MRG_ERROR_BUFFER_OVERFLOW,          /**< buffer overflow */

    /** Errors thrown by the "reader" device */
    MRG_ERROR_TIMEDOUT,                 /**< timeout */
    MRG_ERROR_COLLISION,                /**< collision in air stream */

    /** User configuration errors */
    MRG_ERROR_CONFIG,                   /**< configuration error */
    MRG_ERROR_READER_CONFIG,            /**< reader configuration error */
    MRG_ERROR_ACL_CONFIG,               /**< ACL configuration error */
    MRG_ERROR_LOG_CONFIG,               /**< log configuration error */
    MRG_ERROR_UNDEFINED,                /**< undefined value */

    MRG_ERROR_IO,                       /**< generic IO error */
    MRG_ERROR_EOF,                      /**< end of file */
    MRG_ERROR_IS_DIR,                   /**< file is a directory */
    MRG_ERROR_IS_NOT_DIR,               /**< is not a directory */
    MRG_ERROR_NO_SPACE,                 /**< no space left on device */
    MRG_ERROR_NOT_EMPTY,                /**< Directory not empty */
    MRG_ERROR_NOT_ALLOWED,              /**< Not allowed */
    MRG_ERROR_NOT_EXIST,                /**< Does not exist */
    MRG_ERROR_ALREADY_EXIST,            /**< Already exists */

    MRG_ERROR_SSL,                      /**< Error inherited from SSL library */

    MRG_ERROR_PACKET_OVERFLOW,          /**< Comm layer on top of RFID */

    MRG_ERROR_MALFORMED,                /**< Error in parsing */

    MRG_ERROR_THREAD,                   /**< Generic thread package error. */

    MRG_ERROR_OUT_OF_RANGE,             /**< Generic out of range */

    MRG_ERROR_UNIMPLEMENTED,            /**< Unimplemented */

    MRG_ERROR_N,                        /**<< Max number of ERRNOs */

} mrg_errno_t;


/**
 * For internal usage
 */
void    mrg_error_pre(const char *file, int line);
/**
 * For internal usage
 */
int     mrg_error_print(const char *fmt, ...);

/**
 * Example usage: mrg_error(("This is a forbidden value: %d", x));
 *
 * The double braces are necessary to emulate varargs macros, which stdc doesn't
 * provide.
 */
#define mrg_error(x) \
    do { \
        mrg_error_pre(__FILE__, __LINE__); \
        mrg_error_print x ; \
    } while (0)

/**
 * Print warning
 */
int     mrg_warning(const char *fmt, ...);

/**
 * Reset the global/glocal errno.
 */
int     mrg_error_reset(void);

/** For internal usage */
int     mrg_error_set_annotated(const char *file,
                                int line,
                                int err,
                                const char *msg);
/**
 * Set the global/glocal error value and a message to further specify the
 * error cause.
 */
#define mrg_error_set(err, msg) \
        mrg_error_set_annotated(__FILE__, __LINE__, err, msg)

#ifndef MRG_INLINE
/** If this file is used outside MRG, provide a define for MRG_INLINE */
#  define MRG_INLINE    __inline
#endif
/**
 * toString of #mrg_errno_t
 *
 * The reason this is inlined, is that this function is also used by the
 * stand-alone Guardian Protocol stub compiler, GPc. The tidy way to do this
 * would be to generate #mrg_errno_t and mrg_errno_string()
 * in separate include/src files for the different builds (MRG and GPc). But
 * this file doesn't depend on anything MRG, so can be done the fast&dirty way.
 */
/*@unused@*/
static MRG_INLINE const char *
mrg_errno_string(mrg_errno_t err_code)
{
    switch (err_code) {
    case MRG_ERROR_OK:                  return "OK";

    /* OS errors */
    case MRG_ERROR_RESOURCE_EXHAUSTED:  return "resource exhausted";
    case MRG_ERROR_OUT_OF_MEMORY:       return "out of memory";
    case MRG_ERROR_SCHEDULER:           return "scheduler error";
    case MRG_ERROR_MUTEX:               return "mutex error";

    /* Interrupt errors */
    case MRG_ERROR_INVALID_INTERRUPT:   return "invalid interrupt";
    case MRG_ERROR_NESTED_INTERRUPT:    return "nested interrupt";

    /* Protocol errors */
    case MRG_ERROR_NO_BUF:              return "raw buffer error";
    case MRG_ERROR_NO_SUCH_COMMAND:     return "undefined command value";
    case MRG_ERROR_NO_SUCH_PROTOCOL:    return "no such protocol";
    case MRG_ERROR_PROTOCOL_UNSUPPORTED:return "protocol unsupported";
    case MRG_ERROR_PROTOCOL:            return "protocol error";
    case MRG_ERROR_MANCHESTER_CODING:   return "manchester coding error";
    case MRG_ERROR_AIR_STREAM_CORRUPTED:return "air stream corrupted";
    case MRG_ERROR_INVALID_SYMBOL:      return "invalid symbol";

    /* Driver state */
    case MRG_ERROR_INVALID_STATE:       return "invalid state";
    case MRG_ERROR_EARLY_ABORT:         return "early abort";

    /* Device properties */
    case MRG_ERROR_DEVICE_TOO_SLOW:     return "device too slow";

    /* Generic frame errors */
    case MRG_ERROR_CHECKSUM_FAILS:      return "checksum error";
    case MRG_ERROR_PARITY:              return "parity bit error";

    /* Port handling errors */
    case MRG_ERROR_INTERRUPTED:         return "interrupted blocking call";
    case MRG_ERROR_BUFFER_OVERFLOW:     return "buffer overflow";

    /* Errors thrown by the "reader" device */
    case MRG_ERROR_TIMEDOUT:            return "timed out";
    case MRG_ERROR_COLLISION:           return "RFID response collision";

    /* User configuration errors */
    case MRG_ERROR_CONFIG:              return "configuration error";
    case MRG_ERROR_READER_CONFIG:       return "reader configuration error";
    case MRG_ERROR_ACL_CONFIG:          return "ACL configuration error";
    case MRG_ERROR_LOG_CONFIG:          return "log configuration error";
    case MRG_ERROR_UNDEFINED:           return "undefined";

    /* Generic IO error.
     * error string describes the reason from errno. */
    case MRG_ERROR_IO:                  return "IO error";
    case MRG_ERROR_EOF:                 return "EOF";
    case MRG_ERROR_IS_DIR:              return "is a directory";
    case MRG_ERROR_IS_NOT_DIR:          return "is not a directory";
    case MRG_ERROR_NOT_EMPTY:           return "directory not empty";
    case MRG_ERROR_NOT_ALLOWED:         return "not allowed";
    case MRG_ERROR_NO_SPACE:            return "no space left on device";
    case MRG_ERROR_NOT_EXIST:           return "(component) does not exist";
    case MRG_ERROR_ALREADY_EXIST:       return "already exists";

    /* Error inherited from SSL library */
    case MRG_ERROR_SSL:                 return "SSL error";

    /* Comm layer on top of RFID */
    case MRG_ERROR_PACKET_OVERFLOW:     return "packet overflow";

    /* Error in parsing */
    case MRG_ERROR_MALFORMED:           return "malformed";

    /* Generic thread package error.
     * error string describes the reason from errno. */
    case MRG_ERROR_THREAD:              return "thread error";

    /* Generic out of range */
    case MRG_ERROR_OUT_OF_RANGE:        return "out of range";

    /* Unimplemented */
    case MRG_ERROR_UNIMPLEMENTED:       return "unimplemented";

    /* Total number of errnos */
    case MRG_ERROR_N:                   return "number of errnos";
    }

    return NULL;
}

/**
 * Retrieve the currently pending error value
 */
int mrg_error_get(void);

/**
 * Utility function: mrg_errno_string(mrg_error_get())
 */
const char *mrg_error_string(void);

/**
 * Retrieve the currently pending error description string
 */
const char *mrg_error_describe(void);

#endif


