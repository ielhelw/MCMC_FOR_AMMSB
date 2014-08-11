/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_NETSCIENCE_H__
#define MCMC_PREPROCESS_NETSCIENCE_H__

#include <fstream>

#include <tinyxml.h>

#include "data.h"
#include "preprocess/dataset.h"

namespace mcmc::preprocess {

typedef std::unordered_set<int, std::string> Vertex;

class NetScience : public DataSet<Vertex> {
public:
	NetScience(const char *filename = "datasets/netscience.xml") : filename(filename) {
	}

	virtual ~NetScience() {
	}

	/**
	 * The netscience data is stored in xml format. The function just reads all the vertices
	 * and edges.
	 * * if vertices are not record as the format of 0,1,2,3....., we need to do some 
	 * process.  Fortunally, there is no such issue with netscience data set.   
	 */
	virtual const *mcmc::DATA process() {

		// V stores the mapping between node ID and attribute. i.e title, name. etc
		// i.e {0: "WU, C", 1 :CHUA, L"}
		Vertex *V = new Vertex();
		XMLDocument tree;
		if (tree.LoadFile(filename.c_str()) != XML_NO_ERROR) {
			throw mcmc::IOException("Cannot open " + filename);
		}

		/*
		 * The netscience XML structure is thus:
			<DynamicNetwork>
				<MetaNetwork>
					<documents>
						<document>
					</documents>
					<nodeclass id="agent" type="agent">
						<nodes>
							<node id="0" title="WU, C">
							<node id="1" title="CHUA, L">
							...
						</nodes>
					</nodeclass>
					<networks>
						<network id="agent x agent" isDirected="false">
							<link source="1" target="0" type="double" value="2.5">
							<link source="3" target="2" type="double" value="0.x25">
							...
						</network>
					</networks>
				</MetaNetwork>
			</DynamicNetwork>
		 */
		XMLNode *c;
		c = tree.FirstChildElement("DynamicNetwork");
		if (c == NULL) {
			throw XMLException("Cannot get 'DynamicNetwork'");
		}
		c = c->FirstChildElement("MetaNetwork");
		if (c == NULL) {
			throw XMLException("Cannot get 'MetaNetwork'");
		}
		c = c->FirstChildElement("nodeclass");
		if (c == NULL) {
			throw XMLException("Cannot get 'nodeclass'");
		}
		c = c->FirstChildElement("nodes");
		if (c == NULL) {
			throw XMLException("Cannot get 'nodes'");
		}
		for (XMLElement *n = c->FirstChildElement("node");
				 n != NULL;
				 n = n->NextSiblingElement("node")) {
			XMLElement *id = n->FirstChildElement("id");
			XMLElement *title = n->FirstChildElement("title");
			try {
				(*V)[boost::lexical_cast<int>(id->getText())] = std::string(title->getText());
			} catch (boost::bad_lexical_cast &e) {
				throw mcmc::IOException("Bad number cast from id '" + std::string(id->getText()) + "'");
			}
		}
		
		::size_t N = V->size();
		// iterate every link in the graph, and store those links into Set<Edge> object.
		mcmc::EdgeSet *E = new mcmc::EdgeSet();

		c = tree.FirstChildElement("DynamicNetwork");
		if (c == NULL) {
			throw XMLException("Cannot get 'DynamicNetwork'");
		}
		c = c->FirstChildElement("MetaNetwork");
		if (c == NULL) {
			throw XMLException("Cannot get 'MetaNetwork'");
		}
		c = c->FirstChildElement("networks");
		if (c == NULL) {
			throw XMLException("Cannot get 'networks'");
		}
		c = c->FirstChildElement("network");
		if (c == NULL) {
			throw XMLException("Cannot get 'network'");
		}
		for (XMLElement *n = c->FirstChildElement("link");
				 n != NULL;
				 n = n->NextSiblingElement("link")) {
			XMLElement *source = n->FirstChildElement("source");
			XMLElement *target = n->FirstChildElement("target");
			int a;
			int b;
			try {
				a = boost::lexical_cast<int>(source->getText());
			} catch (boost::bad_lexical_cast &e) {
				throw mcmc::IOException("Bad number cast from source '" + std::string(source->getText()) + "'");
			}
			try {
				b = boost::lexical_cast<int>(target->getText());
			} catch (boost::bad_lexical_cast &e) {
				throw mcmc::IOException("Bad number cast from target '" + std::string(target->getText()) + "'");
			}
			E->insert(mcmc::Edge(a, b));
		}

		return new mcmc::Data<Vertex>(V, E, N);
	}

protected:
	std::string filename;
};

#endif	// ndef MCMC_PREPROCESS_NETSCIENCE_H__
