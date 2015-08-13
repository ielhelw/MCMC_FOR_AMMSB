#ifndef __DAS_LIB_DAS_TIME_H__
#define __DAS_LIB_DAS_TIME_H__

// #ifdef __GNUC__

#include <inttypes.h>

typedef uint64_t	das_time_t, *das_time_p;

#define DAS_TIME_RDTSC(res_p) \
  do { \
    uint32_t eax, edx; \
    __asm __volatile("rdtsc" : "=a" (eax), "=d" (edx)); \
    *res_p = ((uint64_t)edx << 32) + eax; \
  } while (0)

static inline void
das_time_get(das_time_p t)
{
    DAS_TIME_RDTSC(t);
}

double das_time_t2d(const das_time_p t);
void   das_time_d2t(das_time_p t, double d);

void   das_time_init(int *argc, char **argv);
void   das_time_end(void);

// #else
// #error Need GNU C for asm and long long
// #endif

#endif
