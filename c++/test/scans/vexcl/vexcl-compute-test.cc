#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/scan.hpp>

#include <opencl/context.h>

using namespace std;

namespace po = ::boost::program_options;

template <typename T>
double test(const std::string &platformHint, const std::string &deviceHint, size_t len, size_t iters) {

	cl::ClContext clContext = cl::ClContext::createOpenCLContext(platformHint, deviceHint);

	std::vector<T> x(len, static_cast<T>(1));
	cl::Buffer xBuffer(clContext.getContext(), CL_MEM_READ_WRITE, len * sizeof(T));
	clContext.getQueue().enqueueWriteBuffer(xBuffer, CL_TRUE, 0, len * sizeof(T), x.data());
	std::vector<T> y(len);
	cl::Buffer yBuffer(clContext.getContext(), CL_MEM_READ_WRITE, len * sizeof(T));

	// vex::command_queue queue = *(vex::command_queue *)&clQueue;
	vex::Context context(std::vector<cl::Context>(1, clContext.getContext()),
						 std::vector<cl::CommandQueue>(1, clContext.getQueue()));

	vex::backend::opencl::device_vector<T> xDev(xBuffer);
	vex::vector<T> X(context.queue(0), xDev);
	vex::backend::opencl::device_vector<T> yDev(yBuffer);
	vex::vector<T> Y(context.queue(0), yDev);

	double sum = 0;

	for (int i = 0; i < iters; ++i) {
		auto t1 = chrono::system_clock::now();
		vex::inclusive_scan(X, Y);
		clContext.getQueue().finish();
		auto t2 = chrono::system_clock::now();
		chrono::duration<double> duration = t2-t1;
		sum += duration.count();
	}

	clContext.getQueue().enqueueReadBuffer(yBuffer, CL_TRUE, 0, len * sizeof(T), y.data());
	T value = static_cast<T>(1);
	for (size_t i = 0; i < len; i++) {
		if (y[i] != (i + 1) * value) {
			cerr << "Incorrect[" << i << "]: expect " << ((i + 1) * value) << " have " << y[i] << endl;
			break;
		}
	}

	return sum/iters;
}

int main(int argc, char *argv[]) {
	std::string openClPlatform;
	std::string openClDevice;

	size_t len;
	size_t iters;
	
	po::options_description desc("Options");
	desc.add_options()
		("help,?", "help")

		("platform,p", po::value<std::string>(&openClPlatform), "OpenCL platform")
		("device,d", po::value<std::string>(&openClDevice), "OpenCL device")
		("iterations,x", po::value<size_t>(&iters)->default_value(iters), "iterations")
		("items,l", po::value<size_t>(&len)->default_value(6 * 1024 * 1024), "items")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0) {
		std::cout << desc << std::endl;
	}

	cout << "    Device: " << openClDevice << ": " << test<double>(openClPlatform, openClDevice, len, iters) << " seconds" << endl;

	return 0;
}
