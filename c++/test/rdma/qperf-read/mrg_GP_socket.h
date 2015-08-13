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

#ifndef MRG_GP_SOCKET_H__
#define MRG_GP_SOCKET_H__

/**
 * @file GP/mrg_GP_socket.h
 *
 * @brief GP over a socket connection
 *
 * @author 2007 Rutger Hofman, VU Amsterdam
 */

#include <stddef.h>

/**
 * Utility function for easily setting up a socket connection
 */
int mrg_socket_server(char *name, size_t name_length);

/**
 * Utility function for easily setting up a socket connection
 */
int mrg_socket_accept(int server_fd);

/**
 * Utility function for easily setting up a socket connection
 */
int mrg_socket_client(const char *server_name);


#endif

