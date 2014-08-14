#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;

int main(int argc, char *argv[]) {

	Options(argc, argv);

	DataFactory df("netscience");

	df.get_data();
	df.dump_data();

	return 0;
}
