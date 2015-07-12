#ifndef MCMC_FILEIO_H__
#define MCMC_FILEIO_H__

#include <cstdio>
#include <unistd.h>

namespace mcmc {

class FileHandle {
public:
	FileHandle(const std::string& filename, bool compressed, const std::string& mode)
			: compressed_(compressed) {
		if (compressed) {
			std::string cmd;
			if (mode == "r") {
				cmd = std::string("zcat ") + filename;
			} else {
				cmd = std::string("gzip > ") + filename;
			}
			handle_ = popen(cmd.c_str(), mode.c_str());
			if (handle_ == NULL) {
				throw mcmc::MCMCException("Cannot popen(" + cmd + ")");
			}
		} else {
			handle_ = fopen(filename.c_str(), mode.c_str());
			if (handle_ == NULL) {
				throw mcmc::MCMCException("Cannot fopen(" + filename + ")");
			}
		}
	}


	~FileHandle() {
		if (compressed_) {
			pclose(handle_);
		} else {
			fclose(handle_);
		}
	}


	FILE *handle() const {
		return handle_;
	}


	void read_fully(void *v_data, ::size_t size) const {
		char *data = static_cast<char *>(v_data);
		::size_t rd = 0;
		while (rd < size) {
			::size_t r = fread(data + rd, 1, size - rd, handle_);
			if (r == 0) {
				throw mcmc::MCMCException("Cannot fread()");
			}
			rd += r;
		}
	}


	void write_fully(void *v_data, ::size_t size) const {
		char *data = static_cast<char *>(v_data);
		::size_t rd = 0;
		while (rd < size) {
			::size_t r = fwrite(data + rd, 1, size - rd, handle_);
			if (r == 0) {
				throw mcmc::MCMCException("Cannot fwrite()");
			}
			rd += r;
		}
	}

private:
	bool compressed_;
	FILE *handle_ = NULL;
};

}

#endif	// ndef MCMC_FILEIO_H__
