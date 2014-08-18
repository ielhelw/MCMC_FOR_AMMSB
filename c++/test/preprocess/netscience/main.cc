#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;

int main(int argc, char *argv[]) {

	Options options(argc, argv);

	DataFactory df("netscience", options.filename);

	const Data *data = df.get_data();
	data->dump_data();
	delete const_cast<Data *>(data);

	return 0;
}
