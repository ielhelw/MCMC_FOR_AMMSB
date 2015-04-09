/*
 * fileio.h
 *
 *  Created on: Mar 6, 2012
 *      Author: ielhelw
 */

#ifndef MR_FILEIO_H_
#define MR_FILEIO_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>
#include <string>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/algorithm/string/predicate.hpp>
#pragma GCC diagnostic pop

#ifdef HAVE_HADOOP
#include <hdfs.h>
#endif

#ifdef HAVE_S3
#include <wsconn.h>
#endif

#include "mr/io.h"
#include "mcmc/exception.h"

namespace mr {


namespace FILESYSTEM {
	enum Type {
		REGULAR,
#ifdef HAVE_HADOOP
		HDFS,
#endif
#ifdef HAVE_S3
		S3,
#endif
		DEFAULT,
	};
};


class FileInfo {
public:
	FileInfo(const std::string &filename) : filename(filename), replication(0), fileSize(0), blockSize(0) {
	}

	// default destructor is OK

	/**
	 * @param replication -1 means unknown replication (as often as not total
	 * 		replication)
	 */
	void init(int replication,
			  bool isDirectory,
			  off_t fileSize,
			  off_t blockSize) {
		this->replication = replication;
		this->isDirectory = isDirectory;
		this->fileSize = fileSize;
		this->blockSize = blockSize;
	}

	const FileInfo *getFileInfo() const {
		return this;
	}

	::size_t numBlocks() const {
		if (blockSize == 0) {
			throw mcmc::InvalidArgumentException("blockSize appears uninitialized");
		}
		return (fileSize + blockSize - 1) / blockSize;
	}

	/**
	 * @return [""] if replication is unknown (as often as not that
	 * 		means total replication)
	 */
	const std::vector<std::string> *getHosts(::size_t block) const {
		return &hostsAtBlock[block];
	}

	std::string filename;
	int			replication;
	bool		isDirectory;
	off_t		fileSize;
	off_t		blockSize;
	std::vector<std::vector<std::string> > hostsAtBlock;
};


class FileReader : public ReaderInterface {
public:
	FileReader(const std::string &filename)
			: filename(filename), fileSize(0),
			  totalRead(0), position(0) {
	}

	virtual ~FileReader() {
	}

	/**
	 * Must correctly maintain totalRead.
	 * If readv() is overridden, then it must also correctly maintain totalRead.
	 */
	virtual ::size_t read(void *data, ::size_t count);

	virtual bool hasMore();

	virtual ::ssize_t getSize() const {
		return fileSize;
	}

protected:
	virtual ::size_t doRead(void *data, ::size_t n) = 0;

	std::string filename;

	::size_t fileSize;

	// FIXME totalRead and position duplicate functionality. Refactor.
	::size_t totalRead;
	::size_t position;
};


class FileWriter : public WriterInterface {
public:
	FileWriter(const std::string &filename)
			: filename(filename) {
	}

	virtual ~FileWriter() {
	}

protected:
	std::string filename;
};


class FileSystem {
public:
	virtual FileReader *createReader(const std::string &filename) = 0;
	/**
	 * @param flags
	 * 		Implicit, flags = 0: O_CREAT and O_WRONLY
	 * 		Optional: O_TRUNC and O_APPEND
	 */
	virtual FileWriter *createWriter(const std::string &filename,
									 int flags = 0) = 0;

	/**
	 * Create dirname and all path components towards it
	 */
	virtual void mkdir(const std::string &dirname) = 0;

	/**
	 * For a FileInfo with valid filename, get its info. No need to create
	 * a reader or writer to just get info.
	 */
	virtual void getFileInfo(FileInfo &fileInfo) = 0;

	virtual std::vector<std::string> listDirectory(const std::string &dirname) = 0;

	virtual void close() = 0;

	virtual void unlink(const std::string &filename) = 0;

	static FileSystem *openFileSystem(FILESYSTEM::Type type);
};


class FdFileReader : public FileReader {
public:
	FdFileReader(const std::string &filename);

	virtual ~FdFileReader();

	virtual ::size_t readv(const struct ::iovec *iov, int iovcnt);

	virtual void close();

protected:
	virtual ::size_t doRead(void *data, ::size_t n);

    int		fd;

    struct stat st;
};


class FdFileWriter : public FileWriter {
public:
	FdFileWriter(const std::string &filename, int flags = 0, int mode = S_IRUSR | S_IWUSR);

	virtual ~FdFileWriter();

	virtual void write(const void *data, size_t count);

	virtual void writev(const struct iovec *iov, int iovlen);

	virtual void close();

protected:
	int		fd;

    struct stat st;
};


class FdFileSystem : public FileSystem {
public:
	virtual ~FdFileSystem() {
	}

	virtual FileReader *createReader(const std::string &filename);

	virtual FileWriter *createWriter(const std::string &filename,
									 int flags = 0);

	virtual void mkdir(const std::string &dirname);

	virtual void getFileInfo(FileInfo &fileInfo);

	virtual std::vector<std::string> listDirectory(const std::string &dirname);

	virtual void close();

	virtual void unlink(const std::string &filename);
};


#ifdef HAVE_HADOOP

class HDFSFileReader : public FileReader {

public:
	HDFSFileReader(hdfsFS fs, const std::string &filename);

	virtual ~HDFSFileReader();

	virtual void close();

protected:
	virtual ::size_t doRead(void *data, ::size_t n);

	hdfsFS fs;
	hdfsFile file;
};


class HDFSFileWriter : public FileWriter {
public:
	HDFSFileWriter(hdfsFS fs,
				   const std::string &filename,
				   int flags = O_WRONLY);

	virtual ~HDFSFileWriter();

	virtual void write(const void *data, ::size_t count);

	virtual void writev(const struct ::iovec *iov, int iovcnt);

	virtual void close();

protected:
	hdfsFS fs;
	hdfsFile file;
};

class HdfsFileSystem : public FileSystem {
public:
	HdfsFileSystem(const std::string &master = "default", tPort port = 0);

	virtual ~HdfsFileSystem() {
	}

	virtual FileReader *createReader(const std::string &filename);

	virtual FileWriter *createWriter(const std::string &filename,
									 int flags = 0);

	virtual void mkdir(const std::string &dirname);

	virtual std::vector<std::string> listDirectory(const std::string &dirname);

	virtual void getFileInfo(FileInfo &fileInfo);

	virtual void close();

	virtual void unlink(const std::string &filename);

	void upRefCount();

	int getRefCount();

protected:
	hdfsFS	hdfsFs;
	int		refCount;
	std::string master;
};

#endif

#ifdef HAVE_S3

class S3FileSystem : public FileSystem {
public:
	static const std::string ENV_AWS_ACCESS_KEY;
	static const std::string ENV_AWS_SECRET_KEY;

	S3FileSystem();

	S3FileSystem(const S3FileSystem& fs);

	virtual ~S3FileSystem();

	virtual FileReader *createReader(const std::string &filename);

	virtual FileWriter *createWriter(const std::string &filename,
									 int flags = 0);

	virtual void mkdir(const std::string &dirname);


	virtual std::vector<std::string> listDirectory(const std::string &dirname);

	virtual void getFileInfo(FileInfo &fileInfo);

	virtual void close() {}

	virtual void unlink(const std::string &filename) {
		throw UnimplementedException("S3FileSystem::unlink()");
	}

	static std::string getBucketName(const std::string &s) {
		size_t pos = s.find_first_of('/');
		if (pos == std::string::npos) {
			return s;
		} else {
			return s.substr(0, pos);
		}
	}

	static std::string getPath(const std::string& s) {
		size_t pos = s.find_first_of('/');
		if (pos == std::string::npos) {
			return "";
		} else {
			return s.substr(pos+1, s.length());
		}
	}

protected:

	std::string accessKey;
	std::string secretKey;
	webstor::WsConfig config;

	friend class S3FileLoader;
	friend class S3FileWriter;
};

class S3FileLoader : public webstor::WsGetResponseLoader {
public:
	S3FileLoader(S3FileSystem *fs, std::string bucket, std::string key, size_t size);
	virtual ~S3FileLoader();

	virtual size_t  onLoad(const void *chunkData, size_t chunkSize, size_t totalSizeHint);

protected:

	void startLoading();

	webstor::WsConfig s3Config;
	std::string bucket, key;
	size_t fileSize;
	std::string tmpFileName;
	int fd;
	volatile size_t written;
	boost::thread *thread;
	boost::mutex mutex;
	boost::condition_variable cond;

	friend class S3FileReader;
};

class S3FileReader : public FileReader {

public:
	S3FileReader(S3FileSystem* fs, const std::string &filename);

	virtual ~S3FileReader();

	virtual void close();

protected:

	virtual ::size_t doRead(void *data, ::size_t n);

	std::string bucket;
	std::string keyName;
	S3FileLoader *loader;
};

class S3FileWriter : public FdFileWriter {
public:
	S3FileWriter(S3FileSystem* fs,
				   const std::string &filename);

	virtual ~S3FileWriter();

	virtual void close();

protected:

	bool upload(const std::string& bucket, const std::string& key, int fd, struct stat& st);

	std::string s3Filename;
	webstor::WsConfig s3Config;
};

#endif

}		// namespace mr

#endif /* MR_FILEIO_H_ */
