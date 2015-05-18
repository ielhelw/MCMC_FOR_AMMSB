/*
 * Copyright 1997 Vrije Universiteit, The Netherlands.
 * For full copyright and restrictions on use see the file COPYRIGHT in the
 * top level of the Panda distribution.
 */

#ifndef __SYS_PAN_TIMER_H__
#define __SYS_PAN_TIMER_H__


typedef struct PAN_TIMER_T pan_timer_t, *pan_timer_p;

#include <das_time.h>

typedef struct PAN_TIMER_T {
    int        stops;
    int        starts;
    das_time_t total;
    das_time_t start;
    das_time_t stop;
    das_time_t max;
    das_time_t min;
} pan_timer_t;

#define das_time_cmp(a, b)      (*(a) - *(b))
#define das_time_sub(a, b)      do { *(a) -= *(b); } while (0)
#define das_time_add(a, b)      do { *(a) += *(b); } while (0)

#define pan_timer_start(r) \
	do { \
	    if ((r)->starts == (r)->stops) \
		++(r)->starts; \
	    das_time_get(&(r)->start); \
	} while (0)

#ifdef PAN_TIMER_MINMAX
#define pan_timer_stop(r) \
	do { \
	    if ((r)->stops == (r)->starts - 1) { \
		++(r)->stops; \
		das_time_get(&(r)->stop); \
		if (das_time_cmp(&(r)->stop, &(r)->start) > 0) { \
		    das_time_sub(&(r)->stop, &(r)->start); \
		    das_time_add(&(r)->total, &(r)->stop); \
		} \
		if (das_time_cmp(&(r)->stop, &(r)->max) > 0) { \
		    das_time_copy&((r)->max, &(r)->stop); \
		} \
		if (das_time_cmp(&(r)->stop, &(r)->min) < 0) { \
		    das_time_copy(&(r)->min, &(r)->stop); \
		} \
	    } \
	} while (0)
#else
#define pan_timer_stop(r) \
	do { \
	    if ((r)->stops == (r)->starts - 1) { \
		++(r)->stops; \
		das_time_get(&(r)->stop); \
		if ((r)->stop > (r)->start) { \
		    (r)->stop -= (r)->start; \
		    (r)->total += (r)->stop; \
		} \
	    } \
	} while (0)
#endif

#endif
