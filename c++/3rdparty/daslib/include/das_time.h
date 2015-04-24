#ifndef __DAS_LIB_DAS_TIME_H__
#define __DAS_LIB_DAS_TIME_H__

#ifdef __GNUC__

typedef unsigned long long	das_time_t, *das_time_p;

#define DAS_TIME_RDTSC(res_p)					\
    __asm __volatile(						\
		       ".byte 0xf; .byte 0x31"			\
		      : "=A" (*(res_p)))

static inline void
das_time_get(das_time_p t)
{
    DAS_TIME_RDTSC(t);
}

double das_time_t2d(const das_time_p t);
void   das_time_d2t(das_time_p t, double d);

void   das_time_init(int *argc, char **argv);
void   das_time_end(void);

#else
#error Need GNU C for asm and long long
#endif

#endif
