/*
 * fileio.cc
 *
 * TODO FIXME this file should be renamed file-io.cc
 *
 *  Created on: Mar 6, 2012
 *      Author: ielhelw
 */

#include <vector>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#endif
#include <boost/filesystem.hpp>
#ifdef HAVE_S3
#include <boost/thread.hpp>
#endif
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif

#ifdef HAVE_HADOOP
#  include <hdfs.h>
#  include <tinyxml2.h>
#  include <stdint.h>
#endif

#ifdef HAVE_S3
#include <sys/mman.h>
#endif

#include "mcmc/exception.h"
// #include "mr/util.h"
#include "mr/fileio.h"
// #include "mr/log.h"
// #include "mr/interdata.h"

using std::vector;
using std::string;
using mcmc::FileException;
using mcmc::InvalidArgumentException;


namespace mr {


/**
 *
 * Class FileReader
 *
 */

::size_t FileReader::read(void *data, ::size_t count) {
	::size_t rd = doRead(data, count);

	totalRead += rd;

	return rd;
}


bool FileReader::hasMore() {
	return totalRead < static_cast< ::size_t>(getSize());
}


/**
 *
 * Class FdFileReader
 *
 */

FdFileReader::FdFileReader(const string &filename)
		: FileReader(filename) {
	fd = ::open(filename.c_str(), O_RDONLY);
	if (fd < 0) {
		throw FileException("open(\"" + filename + "\") fails");
	}

	struct stat st;
	if (fstat(fd, &st) == -1) {
		throw FileException("fstat(\"" + filename + "\") fails");
	}
	fileSize = st.st_size;
}


FdFileReader::~FdFileReader() {
	close();
}


::size_t FdFileReader::readv(const struct ::iovec *iov, int iovcnt) {
	::size_t rd;

	::ssize_t rds = ::readv(fd, iov, iovcnt);
	if (rds == -1) {
		throw FileException("readv() fails");
	}

	rd = static_cast< ::size_t>(rds);

	totalRead += rd;

	return rd;
}


void FdFileReader::close() {
	if (fd != -1) {
		::close(fd);
		fd = -1;
	}
}


::size_t FdFileReader::doRead(void *data, ::size_t n) {
	::ssize_t rd = ::read(fd, data, n);
	if (rd == -1) {
		throw FileException("read() fails");
	}

	return static_cast< ::size_t>(rd);
}


#ifdef HAVE_HADOOP

/**
 *
 * class HDFSFileReader
 *
 */

HDFSFileReader::HDFSFileReader(hdfsFS fs, const string &filename)
		: FileReader(filename),
		  fs(fs), file(NULL) {
	file = hdfsOpenFile(fs, filename.c_str(), O_RDONLY, 0, 0, 0);
	if (file == NULL) {
		throw FileException(string("hdfsOpenFile(") + filename +
								string(") fails"));
	}

	hdfsFileInfo *info = hdfsGetPathInfo(fs, filename.c_str());
	fileSize = info->mSize;
	hdfsFreeFileInfo(info, 1);
}


HDFSFileReader::~HDFSFileReader() {
	close();
}


void HDFSFileReader::close() {
	if (file != NULL) {
		if (hdfsCloseFile(fs, file) != 0) {
			throw FileException("hdfsCloseFile() fails");
		}
		file = NULL;
	}
}


::size_t HDFSFileReader::doRead(void *data, ::size_t n) {
	tSize totalRd = 0;
	while (totalRd < (tSize)n && position < fileSize) {
		tSize rd = hdfsRead(fs, file, (char *)data + totalRd, (tSize)n - totalRd);
		if (rd == -1) {
			throw FileException("read() fails");
		}
		position += rd;
		totalRd += rd;
	}

	return totalRd;
}

#endif


/**
 *
 * class FdFileWriter
 *
 */


void mkdir_p(const char *dir) {
	boost::filesystem::create_directories(dir);
}


FdFileWriter::FdFileWriter(const string &filename, int flags, int mode)
		: FileWriter(filename) {
	assert((flags & O_TRUNC) | (flags & O_APPEND));
	flags |= O_WRONLY | O_CREAT;

	boost::filesystem::path path(filename);
	boost::filesystem::path dirname = path.parent_path();
	mkdir_p(dirname.string().c_str());

	fd = ::open(filename.c_str(), flags, mode);
	if (fd < 0) {
		throw FileException(string("open(") + filename + string(") fails"));
	}
	if (::fstat(fd, &st)) {
		throw FileException(string("fstat") + filename + string(") fails"));
	}
}


FdFileWriter::~FdFileWriter() {
	close();
}


void FdFileWriter::write(const void *data, size_t count) {
	::size_t total = 0;
	while (total < count) {
		::ssize_t wr = ::write(fd, (const char *)data + total, count - total);
		if (wr == -1) {
			if (errno == EINTR) {
				errno = 0;
			} else {
				throw FileException("write() fails");
			}
		}
		total += wr;
	}
}


void FdFileWriter::writev(const struct iovec *iov_orig, int iovlen_orig) {
	int iovlen = iovlen_orig;
	struct iovec iov[iovlen];
	::size_t count = 0;
	for (int i = 0; i < iovlen; i++) {
		iov[i] = iov_orig[i];
		count += iov[i].iov_len;
	}

	::size_t total = 0;
	struct iovec *iov_current = iov;
	while (total < count) {
		::ssize_t w = ::writev(fd, iov_current, iovlen);
		if (w == -1) {
			if (errno == EINTR) {
				errno = 0;
			} else {
				throw FileException("writev() fails");
			}
		} else {
			total += w;
			while (true) {
				if ((::ssize_t)iov_current[0].iov_len > w) {
					iov_current[0].iov_len -= w;
					iov_current[0].iov_base = (char *)iov_current[0].iov_base + w;
					break;
				} else {
					w -= iov_current[0].iov_len;
					iov_current++;
				}
			}
		}
	}
}


void FdFileWriter::close() {
	if (fd != -1) {
		::close(fd);
		fd = -1;
	}
}


#ifdef HAVE_HADOOP

/**
 *
 * class HDFSFileWriter
 *
 */

HDFSFileWriter::HDFSFileWriter(hdfsFS fs,
							   const string &filename,
							   int flags)
		: FileWriter(filename), fs(fs) {
	file = hdfsOpenFile(fs, filename.c_str(), flags, 0, 0, 0);
	if (file == NULL) {
		throw FileException(string("hdfsOpenFile(") + filename + string(") fails"));
	}
}


HDFSFileWriter::~HDFSFileWriter() {
	close();
}


void HDFSFileWriter::write(const void *data, ::size_t count) {
	::size_t total = 0;
	while (total < count) {
		tSize wr = hdfsWrite(fs, file,
							 (const char *)data + total,
							 (tSize)(count - total));
		if (wr == -1) {
			throw FileException("hdfsWrite() fails");
		}
		total += wr;
	}
}


void HDFSFileWriter::writev(const struct ::iovec *iov, int iovcnt) {
	WriterInterface::writev(iov, iovcnt);
}


void HDFSFileWriter::close() {
	if (file != NULL) {
		if (hdfsFlush(fs, file) != 0) {
			throw FileException("hdfsFlushFile() fails");
		}
		hdfsCloseFile(fs, file);
		file = NULL;
	}
}

#endif


/**
 *
 * class FdFileSystem
 *
 */

FileReader *FdFileSystem::createReader(const string &filename) {
	return new FdFileReader(filename.c_str());
}


FileWriter *FdFileSystem::createWriter(const string &filename,
									   int flags) {
	return new FdFileWriter(filename.c_str(), O_CREAT | O_WRONLY | flags);
}


void FdFileSystem::mkdir(const string &dirname) {
	mkdir_p(dirname.c_str());
}


void FdFileSystem::getFileInfo(FileInfo &fileInfo) {
	struct stat st;
	if (::stat(fileInfo.filename.c_str(), &st) == -1) {
		throw FileException("fstat(\"" + fileInfo.filename + "\") fails");
	}

	fileInfo.init(-1,
				  (st.st_mode & S_IFDIR) == S_IFDIR,
				  st.st_size,
				  st.st_size);

	vector<string> hostsAtTheBlock = vector<string>(1, "");
	fileInfo.hostsAtBlock = vector<vector<string> >(1, hostsAtTheBlock);
}


vector<string> FdFileSystem::listDirectory(const string &dirname) {
	DIR *d = opendir(dirname.c_str());
	if (d == NULL) {
		throw FileException("opendir(\"" + dirname + "\") fails");
	}

	vector<string> contents;
	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		contents.push_back(dirname + "/" + de->d_name);
	}

	if (errno != 0) {
		throw FileException("readdir(\"" + dirname + "\") fails");
	}

	return contents;
}


void FdFileSystem::close() {
}

void FdFileSystem::unlink(const string &filename) {
	if (::unlink(filename.c_str()) == -1) {
		throw FileException("unlink(\"" + filename + "\") fails");
	}
}

static FdFileSystem *regularFileSystem = NULL;


#ifdef HAVE_HADOOP

using tinyxml2::XMLDocument;
using tinyxml2::XMLNode;
using tinyxml2::XMLElement;
using tinyxml2::XML_NO_ERROR;

static void getHdfsMaster(string &master, tPort &port) {
	globalLog() << "Parse $HADOOP_CONF_DIR/core-site.xml to retrieve value for <configuration> <property> <name = fs.default.name> <value = hdfs://master.ip.addr:port/> </property> </configuration>";
	const char *confDir = getenv("HADOOP_CONF_DIR");
	if (confDir == NULL) {
		master = "default";
		port = 0;

		return;
	}

	string coreSiteFile(confDir);
	coreSiteFile += "/core-site.xml";
	XMLDocument coreSite;            
	if (coreSite.LoadFile(coreSiteFile.c_str()) != XML_NO_ERROR) {
		std::ostringstream o;
		o << "Cannot open " << confDir;
		throw IOException(o.str());
	}                                

	XMLNode *c = coreSite.FirstChildElement("configuration");
	if (c == NULL) {
		throw IOException("Cannot find XML element <configuration>");
	}

	XMLElement *m = NULL;
	XMLElement *p = c->FirstChildElement("property");
	for (; p != NULL; p = p->NextSiblingElement("property")) {
		XMLElement *name = p->FirstChildElement("name");
		if (name != NULL) {
			if (strcmp(name->GetText(), "fs.default.name") == 0) {
				m = p->FirstChildElement("value");
				goto outer;
			}
		}
	}
outer:

	if (m == NULL) {
		throw IOException("Cannot find XML element property->name");
	}

	const char *nodePort = m->GetText();
	const char *hdfsPrefix = "hdfs://";
	if (strncmp(nodePort, hdfsPrefix, strlen(hdfsPrefix)) != 0) {
		throw IOException("nodePort does not have hdfs:// prefix");
	}
	nodePort += strlen(hdfsPrefix);
	const char *col = strchr(nodePort, ':');
	if (col == NULL) {
		throw IOException("Cannot find ':' in nodePort name string");
	}
	master = string(nodePort, col - nodePort);
	if (sscanf(col + 1, "%hu", &port) != 1) {
		throw IOException("Cannot parse ushort value for port");
	}

	globalLog() << "Found master '" << master << "' port " << port;
}


/**
 *
 * class HdfsFileSystem
 *
 */

HdfsFileSystem::HdfsFileSystem(const string &master, tPort port)
		: refCount(0) {
	hdfsFs = hdfsConnect(master.c_str(), port);
	if (hdfsFs == NULL) {
		throw IOException("Cannot connect to HDFS");
	}
   
	std::ostringstream o;
	o << "hdfs://" << master << ":" << port;
	this->master = o.str();
}


void HdfsFileSystem::close() {
	refCount--;
	if (refCount == 0) {
		globalLog() << "Disconnect from hdfs";
		hdfsDisconnect(hdfsFs);
		hdfsFs = NULL;
	}
}

void HdfsFileSystem::unlink(const string &filename) {
	hdfsDelete(hdfsFs, filename.c_str());
}


FileReader *HdfsFileSystem::createReader(const string &filename) {
	return new HDFSFileReader(hdfsFs, filename.c_str());
}


FileWriter *HdfsFileSystem::createWriter(const string &filename,
										 int flags) {
	flags &= ~O_CREAT;	// enabled by default
	if (! (flags & O_TRUNC)) {
		flags |= O_APPEND;
	}

	return new HDFSFileWriter(hdfsFs, filename.c_str(), O_WRONLY | flags);
}


void HdfsFileSystem::mkdir(const string &dirname) {
	hdfsCreateDirectory(hdfsFs, dirname.c_str());
}


void HdfsFileSystem::getFileInfo(FileInfo &fileInfo) {
	hdfsFileInfo *hdfsInfo = hdfsGetPathInfo(hdfsFs,
											 fileInfo.filename.c_str());
	if (hdfsInfo == NULL) {
		throw IOException("Cannot hdfsGetPathInfo(" + fileInfo.filename + ")");
	}

	fileInfo.init(hdfsInfo->mReplication,
				  hdfsInfo->mKind == kObjectKindDirectory,
				  hdfsInfo->mSize,
				  hdfsInfo->mBlockSize);

	hdfsFreeFileInfo(hdfsInfo, 1);

	char ***blockInfo = hdfsGetHosts(hdfsFs,
									 fileInfo.filename.c_str(),
									 0,
									 fileInfo.fileSize);
	if (blockInfo == NULL) {
		throw IOException("Cannot hdfsGetHosts(" + fileInfo.filename + ")");
	}

	if (fileInfo.fileSize == 0) {
		// globalLog() << "What did you think; a newly created file has no blocks anywhere";
	} else {
		int i = 0;
		while (blockInfo[i] != NULL) {
			vector<string> hostsAtOneBlock;
			for (int h = 0; blockInfo[i][h] != NULL && h < fileInfo.replication; h++) {
				hostsAtOneBlock.push_back(string(blockInfo[i][h]));
			}
			fileInfo.hostsAtBlock.push_back(hostsAtOneBlock);

			i++;
		}
	}

	hdfsFreeHosts(blockInfo);
}


vector<string> HdfsFileSystem::listDirectory(const string &dirname) {
	int nEntries;
	hdfsFileInfo *dir = hdfsListDirectory(hdfsFs, dirname.c_str(), &nEntries);
	if (dir == NULL) {
		throw IOException("Cannot hdfsListDirectory(\"" + dirname + "\")");
	}

	vector<string> contents;
	::size_t masterLen = master.size();
	for (int i = 0; i < nEntries; i++) {
		string e(dir[i].mName);
		if (e.compare(0, masterLen, master) == 0) {
			contents.push_back(dir[i].mName + masterLen);
		} else {
			contents.push_back(dir[i].mName);
		}
	}

	hdfsFreeFileInfo(dir, nEntries);

	return contents;
}


void HdfsFileSystem::upRefCount() {
	refCount++;
}

int HdfsFileSystem::getRefCount() {
	return refCount;
}


static HdfsFileSystem *hdfsFileSystem = NULL;


#endif


#ifdef HAVE_S3

const string S3FileSystem::ENV_AWS_ACCESS_KEY = "AWS_ACCESS_KEY";
const string S3FileSystem::ENV_AWS_SECRET_KEY = "AWS_SECRET_KEY";

static S3FileSystem *s3FileSystem = NULL;

S3FileSystem::S3FileSystem() {
	char *p = getenv(ENV_AWS_ACCESS_KEY.c_str());
	if (p == NULL)
		throw std::runtime_error(ENV_AWS_ACCESS_KEY + " was not set in environment");
	accessKey = p;
	p = getenv(ENV_AWS_SECRET_KEY.c_str());
	if (p == NULL)
		throw std::runtime_error(ENV_AWS_SECRET_KEY + " was not set in environment");
	secretKey = p;

	config.accKey = accessKey.c_str();
	config.secKey = secretKey.c_str();
	config.storType = webstor::WST_S3;
	config.host = "s3.amazonaws.com";
	config.isHttps = true;
	config.port = NULL;
	config.proxy = NULL;
	config.sslCertFile = NULL;
}

S3FileSystem::S3FileSystem(const S3FileSystem& fs) {
	accessKey = fs.accessKey;
	secretKey = fs.secretKey;
	config = fs.config;
}

S3FileSystem::~S3FileSystem() {}

void S3FileSystem::mkdir(const string &dirname __attribute__((unused))) {}

vector<string> S3FileSystem::listDirectory(const string &dirname) {
	string bucket = getBucketName(dirname);
	string path = getPath(dirname);
	if (path.length() > 0 && path[path.length()-1] != '/') {
		path += '/';
	}
	webstor::WsConnection conn(config);
	vector<webstor::WsObject> objs;
	conn.listObjects(bucket.c_str(), path.c_str(), NULL, "/", 0, &objs);
	vector<string> vec;
	for (vector<webstor::WsObject>::iterator i = objs.begin();
			i != objs.end(); ++i) {
		vec.push_back(i->key);
	}
	return vec;
}

void S3FileSystem::getFileInfo(FileInfo &fileInfo) {
	string bucket = getBucketName(fileInfo.filename);
	string path = getPath(fileInfo.filename);
	vector<webstor::WsObject> objs;
	webstor::WsConnection conn(config);
	conn.listObjects(bucket.c_str(), path.c_str(), NULL, "/", 0, &objs);
	if (objs.empty()) {
		throw FileException("Failed to find file");
	} else if (objs.size() > 1) {
		throw FileException("Expected singled file, found multiple");
	}
	fileInfo.init(-1, objs[0].isDir, objs[0].size, objs[0].size);
	vector<string> hostsAtTheBlock = vector<string>(1, "");
	fileInfo.hostsAtBlock = vector<vector<string> >(1, hostsAtTheBlock);
}

FileReader *S3FileSystem::createReader(const string &filename) {
	return new S3FileReader(this, filename);
}

FileWriter *S3FileSystem::createWriter(const string &filename __attribute__((unused)),
								 int flags __attribute__((unused))) {
	return new S3FileWriter(this, filename);
}

S3FileLoader::S3FileLoader(S3FileSystem *fs, string bucket, string key, size_t size)
			: s3Config(fs->config), bucket(bucket), key(key), fileSize(size), written(0) {
	tmpFileName = tmpnam(NULL);
	fd = open(tmpFileName.c_str(), O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
	if (fd == -1) {
		std::ostringstream o;
		o << "Failed to open file: " << tmpFileName;
		throw FileException(o.str());
	}
	thread = new boost::thread(boost::bind(&S3FileLoader::startLoading, this));
}

S3FileLoader::~S3FileLoader() {
	thread->join();
	delete thread;
	::close(fd);
	::remove(tmpFileName.c_str());
}

void S3FileLoader::startLoading() {
	int attempts = 4;
	bool failed;
	do {
		failed = false;
		try {
			webstor::WsGetResponse res;
			webstor::WsConnection conn(s3Config);
			conn.get(bucket.c_str(), key.c_str(), this, &res);
			assert(res.loadedContentLength == fileSize);
			assert(!res.isTruncated);
		} catch (std::exception &e) {
			failed = true;
			--attempts;
			globalLog() << "S3FileLoader: failed to get file " << key << ": " << e.what();
			if (attempts == 0) {
				globalLog() << "S3FileLoader: All attempts failed to get " << key;
				throw;
			} else {
				boost::mutex::scoped_lock lock(mutex);
				// reset parameters
				written = 0;
				lseek(fd, 0, SEEK_SET);
			}
		}
	} while(failed && attempts > 0);
}

size_t S3FileLoader::onLoad(const void *chunkData, size_t chunkSize, size_t totalSizeHint __attribute__((unused))) {
	write_full(fd, chunkData, chunkSize);
	{
		boost::mutex::scoped_lock lock(mutex);
		written += chunkSize;
	}
	cond.notify_one();
	return chunkSize;
}

S3FileReader::S3FileReader(S3FileSystem* fs,
			   const string &filename) : FileReader(filename) {
	FileInfo info(filename);
	fs->getFileInfo(info);
	fileSize = info.fileSize;
	bucket = S3FileSystem::getBucketName(filename);
	keyName = S3FileSystem::getPath(filename);
	loader = new S3FileLoader(fs, bucket, keyName, info.fileSize);
}

S3FileReader::~S3FileReader() {
	if (loader != NULL) {
		close();
	}
}

void S3FileReader::close() {
	if (loader != NULL) {
		delete loader;
		loader = NULL;
	}
}

::size_t S3FileReader::doRead(void *data, ::size_t n) {
	size_t toRead = std::min(n, fileSize - position);
	{
		boost::mutex::scoped_lock lock(loader->mutex);
		while (loader->written < position + toRead) {
			loader->cond.wait(lock);
		}
	}

	if (toRead) {
		pread_full(loader->fd, data, toRead, position);
		position += toRead;
	}

	return toRead;
}

S3FileWriter::S3FileWriter(S3FileSystem* fs,
				   const string &filename)
	: FdFileWriter(tmpnam(NULL), O_TRUNC), s3Filename(filename), s3Config(fs->config) {
}

S3FileWriter::~S3FileWriter() {
	if (fd != -1) {
		close();
	}
}

void S3FileWriter::close() {
	if (fd != -1) {
		FdFileWriter::close();

		timer upload_time("FINAL OUTPUT TO S3");
		upload_time.start();
		struct stat st;
		fd = ::open(filename.c_str(), O_RDONLY);
		if (fd < 0) {
			throw FileException(string("open(") + filename + string(") fails"));
		}
		if (::fstat(fd, &st)) {
			throw FileException(string("fstat") + filename + string(") fails"));
		}
		string bucket = S3FileSystem::getBucketName(s3Filename);
		string key = S3FileSystem::getPath(s3Filename);

		int attempts = 4;

		while (attempts > 0 && !upload(bucket, key, fd, st)) {
			--attempts;
		}
		if (attempts == 0) {
			std::ostringstream o;
			o << "S3FileWriter: Failed to upload " << key;
			globalLog() << o.str();
			throw FileException(o.str());
		}

		::close(fd);
		fd = -1;
		remove(filename.c_str());
		upload_time.stop();

		globalLog() << upload_time;
	}
}

bool S3FileWriter::upload(const string& bucket, const string& key, int fd, struct stat& st) {
	static const size_t LIMIT_5MB = 5*1024*1024;
	char* buf = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
	if (buf == MAP_FAILED) {
		throw FileException("Failed to map file");
	}
	bool status = true;
	try {
		webstor::WsConnection conn(s3Config);
		if (st.st_size < (off_t)LIMIT_5MB) {
			conn.put(bucket.c_str(), key.c_str(), buf, st.st_size);
		} else {
			webstor::WsInitiateMultipartUploadResponse res;
			conn.initiateMultipartUpload(bucket.c_str(), key.c_str(), NULL,
					webstor::WsConnection::c_noCacheControl, false, false, &res);
			vector<webstor::WsPutResponse> responses;
			size_t off = 0;
			int i = 1;
			for (; off < (size_t)st.st_size; ++i) {
				static const size_t chunkSize = (1<<24) /* 16MB : should be multiple of vm page size */;
				size_t size = std::min(st.st_size - off, chunkSize);
				webstor::WsPutResponse response;
				conn.putPart(bucket.c_str(), key.c_str(), res.uploadId.c_str(), i, buf + off, size, &response);
				responses.push_back(response);
				munmap(buf + off, size);
				off += size;
			}
			conn.completeMultipartUpload(bucket.c_str(), key.c_str(), res.uploadId.c_str(), &responses[0], responses.size());
		}
	} catch (std::exception &e) {
		status = false;
		globalLog() << "S3FileWriter: failed to upload file " << key << ": " << e.what();
	}
	munmap(buf, st.st_size);
	return status;
}

#endif

/**
 *
 * class FileSystem
 *
 */


FileSystem *FileSystem::openFileSystem(FILESYSTEM::Type type) {
	switch (type) {
#ifdef HAVE_HADOOP
	case FILESYSTEM::HDFS:
		globalLog() << "In openFileSystem(HDFS): current " << hdfsFileSystem;
		if (hdfsFileSystem == NULL || hdfsFileSystem->getRefCount() == 0) {
			if (hdfsFileSystem != NULL) {
				delete hdfsFileSystem;
			}
			globalLog() << "HADOOP_CONF_DIR=" << getenv("HADOOP_CONF_DIR");
			string master;
			tPort port;
			getHdfsMaster(master, port);
			hdfsFileSystem = new HdfsFileSystem(master, port);
		}
		hdfsFileSystem->upRefCount();
		return hdfsFileSystem;
#endif
#ifdef HAVE_S3
	case FILESYSTEM::S3:
		if (s3FileSystem == NULL) {
			globalLog() << "CREATEING S3 FILESYSTEM";
			s3FileSystem = new S3FileSystem();
		}
		return s3FileSystem;
#endif
	case FILESYSTEM::REGULAR:
		if (regularFileSystem == NULL) {
			regularFileSystem = new FdFileSystem();
		}
		return regularFileSystem;
	default:
		throw InvalidArgumentException("Unknown filesystem type");
		//NOTREACHED
		return NULL;
	}
}

}		// namespace mr
