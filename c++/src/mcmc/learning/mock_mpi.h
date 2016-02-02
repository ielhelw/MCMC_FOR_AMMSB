#ifndef MPI_H_
#define MPI_H_

#include <inttypes.h>

#define MPI_SUCCESS		0

typedef int MPI_Comm;
#define MPI_COMM_WORLD	0

enum MPI_ERRORS {
  MPI_ERRORS_RETURN,
  MPI_ERRORS_ARE_FATAL,
};

enum MPI_Datatype {
  MPI_INT                    = 0x4c000405,
  MPI_LONG                   = 0x4c000407,
  MPI_UNSIGNED_LONG          = 0x4c000408,
  MPI_FLOAT                  = 0x4c00040a,
  MPI_DOUBLE                 = 0x4c00080b,
  MPI_BYTE                   = 0x4c00010d,
};

enum MPI_Op {
  MPI_SUM,
};

void *MPI_IN_PLACE = (void *)0x88888888;


int MPI_Init(int *argc, char ***argv) {
  return MPI_SUCCESS;
}

int MPI_Finalize() {
  return MPI_SUCCESS;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
  return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm) {
  return MPI_SUCCESS;
}

int MPI_Comm_set_errhandler(MPI_Comm comm, int mode) {
  return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int *mpi_size) {
  *mpi_size = 1;
  return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int *mpi_rank) {
  *mpi_rank = 0;
  return MPI_SUCCESS;
}

::size_t mpi_datatype_size(MPI_Datatype type) {
  switch (type) {
  case MPI_INT:
    return sizeof(int32_t);
  case MPI_LONG:
    return sizeof(int64_t);
  case MPI_UNSIGNED_LONG:
    return sizeof(uint64_t);
  case MPI_FLOAT:
    return sizeof(float);
  case MPI_DOUBLE:
    return sizeof(double);
  case MPI_BYTE:
    return 1;
  default:
    std::cerr << "Unknown MPI datatype" << std::cerr;
    return 0;
  }
}

int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
  memcpy(recvbuf, sendbuf, sendcount * mpi_datatype_size(sendtype));
  return MPI_SUCCESS;
}

int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
  return MPI_Scatter((char *)sendbuf + displs[0] * mpi_datatype_size(sendtype),
                     sendcounts[0], sendtype,
                     recvbuf, recvcount, recvtype,
                     root, comm);
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm) {
  memcpy(recvbuf, sendbuf, count * mpi_datatype_size(datatype));
  return MPI_SUCCESS;
}

int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) {
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * mpi_datatype_size(datatype));
  }
  return MPI_SUCCESS;
}

#endif  // ndef MPI_H_
