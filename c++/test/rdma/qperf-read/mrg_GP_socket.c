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


#include "mrg_GP_socket.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/time.h>
#include <sys/select.h>
#include <signal.h>

#include <errno.h>
#include <string.h>
#include <limits.h>

#include "mrg_util.h"
#include "mrg_errno.h"

/**
 * ---------------------------------------------------------------------------
 *
 * Utility functions for easily setting up a socket connection
 *
 * ---------------------------------------------------------------------------
 */

int
mrg_socket_server(char *name, size_t name_length)
{
    socklen_t   size;
#define HOSTNAME_LENGTH         1024
    char        hostname[HOSTNAME_LENGTH];
    int         fd;
    struct sockaddr_in addr;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        return -1;
    }

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        return -1;
    }

    size = sizeof addr;
    if (getsockname(fd, (struct sockaddr *)&addr, &size) == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        return -1;
    }

    if (gethostname(hostname, sizeof hostname) == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        return -1;
    }

    if (snprintf(name,
                 name_length,
                 "%s/%d",
                 hostname,
                 htons(addr.sin_port)) >= (int)name_length) {
        mrg_error_set(MRG_ERROR_BUFFER_OVERFLOW, "hostname/port does not fit");
        return -1;
    }

    if (listen(fd, 5) != 0) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        return -1;
    }

    return fd;
}


int
mrg_socket_accept(int server_fd)
{
    socklen_t   size;
    struct sockaddr_in addr;
    int         fd = -1;

    size = sizeof addr;
    do {
        fd = accept(server_fd, (struct sockaddr *)&addr, &size);
        if (fd == -1) {
            if (errno != EINTR) {
                mrg_error_set(MRG_ERROR_IO, strerror(errno));
                errno = 0;
                return -1;
            }
            errno = 0;
        }
    } while (fd == -1);

    return fd;
}


int
mrg_socket_client(const char *server_name)
{
    char *slash;
    unsigned short port;
    struct hostent *hp;
    socklen_t   size;
    char        name[HOSTNAME_LENGTH];
    int         fd = -1;
    struct sockaddr_in addr;

    addr.sin_family = AF_INET;

    slash = strchr(server_name, '/');
    if (slash == NULL) {
        mrg_error_set(MRG_ERROR_MALFORMED, "does not contain '/'");
        goto error;
    }
    if (slash == server_name) {
        strncpy(name, "localhost", sizeof name);
    } else {
        if (slash - server_name > (int)(sizeof name - 1)) {
            mrg_error_set(MRG_ERROR_BUFFER_OVERFLOW,
                          "name string cannot contain server name");
            goto error;
        }
        strncpy(name, server_name, slash - server_name);
        name[slash - server_name] = '\0';
    }

    if (sscanf(slash + 1, "%hu", &port) != 1) {
        mrg_error_set(MRG_ERROR_MALFORMED, "does not contain port number");
        goto error;
    }

    addr.sin_port = htons(port);
    hp = gethostbyname(name);
    if (hp == NULL) {
#ifdef MRG_WIN32
        if (WSAGetLastError() != 0) {
            if (WSAGetLastError() == 11001) {
                mrg_diag_printf("Host %s not found\n", name);
            } else {
                mrg_diag_printf("gethostbyname error#: %ld\n", WSAGetLastError());
            }
        }
#endif
        mrg_error_set(MRG_ERROR_IO, strerror(h_errno));
        goto error;
    }
    if (hp->h_length > (int)(sizeof addr.sin_addr)) {
        mrg_error_set(MRG_ERROR_BUFFER_OVERFLOW, "inet length does not fit");
        goto error;
    }
    memcpy(&addr.sin_addr, hp->h_addr_list[0], hp->h_length);

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        goto error;
    }

    while (connect(fd, (struct sockaddr *)&addr, sizeof addr) == -1) {
        if (errno == EINTR) {
            errno = 0;
        } else {
            mrg_error_set(MRG_ERROR_IO, strerror(errno));
            errno = 0;
            goto error;
        }
    }

    size = sizeof addr;
    if (getsockname(fd, (struct sockaddr *)&addr, &size) == -1) {
        mrg_error_set(MRG_ERROR_IO, strerror(errno));
        errno = 0;
        goto error;
    }

    return fd;

error:
    if (fd != -1) {
        close(fd);
    }

    return -1;
}
