#ifndef CONTEXT_H_
#define CONTEXT_H_

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif/*__CL_ENABLE_EXCEPTIONS*/

#include <CL/cl.hpp>

namespace cl {

class ClContext {
public:
	static ClContext createOpenCLContext(std::string platformHint = "",
			std::string deviceHint = "",
			cl_command_queue_properties qprops = CL_QUEUE_PROFILING_ENABLE
					| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.empty())
			abort();

		for (auto p : platforms) {
			if (!platformHint.empty()
					&& p.getInfo<CL_PLATFORM_NAME>().find(platformHint)
							== std::string::npos) {
				continue;
			}
			std::vector<cl::Device> devices;
			p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			for (auto d : devices) {
				if (!deviceHint.empty()
						&& d.getInfo<CL_DEVICE_NAME>().find(deviceHint)
								== std::string::npos) {
					continue;
				}
				return ClContext(p, d, qprops);
			}
		}
		throw std::runtime_error(
				"Failed to find specified OpenCL platform/device");
	}

	cl::Program createProgram(const std::string& filename, const std::string& opts = "") {
		std::ifstream f(filename);
		std::string progText((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		cl::Program program(context, cl::Program::Sources(1, std::make_pair(progText.c_str(), progText.length())));
		try {
			program.build(opts.c_str());
		} catch (...) {
			cl_int status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			std::ostringstream out;
			out << "Failed to build program " << filename << " (error code " << status << "). Build log:\n" << log;
			throw std::runtime_error(out.str());
		}
		return program;
	}

protected:
	ClContext(const cl::Platform& platform, const cl::Device& device,
			const cl_command_queue_properties qprops) :
			device(device), platform(platform) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM,
				(cl_context_properties) (platform)(), 0 };
		context = cl::Context(device, props);
		queue = cl::CommandQueue(context, device, qprops);
	}

public:
	cl::Device device;
	cl::Platform platform;
	cl::Context context;
	cl::CommandQueue queue;
};

}

#endif /* CONTEXT_H_ */
