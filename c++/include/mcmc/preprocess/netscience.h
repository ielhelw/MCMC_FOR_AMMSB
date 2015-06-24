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
#include <unordered_map>

#include <tinyxml2.h>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

namespace tinyxml2 {

class XMLException : public std::exception {
public:
	XMLException(const std::string &reason) throw()
			: reason("XMLException" + reason) {
	}

	virtual ~XMLException() throw() {
	}

	virtual const char *what() const throw() {
		return reason.c_str();
	}

protected:
	std::string reason;
};

}	// namespace tinyxml2


namespace mcmc {
namespace preprocess {

using namespace tinyxml2;

typedef std::unordered_map<int, std::string> Vertex;

class NetScience : public DataSet {
public:
	NetScience(const std::string &filename, bool compressed = false,
			   bool contiguous = false)
			: DataSet(filename == "" ? "datasets/netscience.xml" : filename,
					  compressed, contiguous) {
	}

	virtual ~NetScience() {
	}

	/**
	 * The netscience data is stored in xml format. The function just reads all the vertices
	 * and edges.
	 * * if vertices are not record as the format of 0,1,2,3....., we need to do some 
	 * process.  Fortunally, there is no such issue with netscience data set.   
	 */
	virtual const mcmc::Data *process() {

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
					<nodes>
						<nodeclass id="agent" type="agent">
							<node id="0" title="WU, C">
							<node id="1" title="CHUA, L">
							...
						</nodeclass>
					</nodes>
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
		c = c->FirstChildElement("nodes");
		if (c == NULL) {
			throw XMLException("Cannot get 'nodes'");
		}
		c = c->FirstChildElement("nodeclass");
		if (c == NULL) {
			throw XMLException("Cannot get 'nodeclass'");
		}
		for (XMLElement *n = c->FirstChildElement("node");
				 n != NULL;
				 n = n->NextSiblingElement("node")) {
			int id;
			if (n->QueryIntAttribute("id", &id) != XML_NO_ERROR) {
				throw XMLException("Cannot get int attribute 'id'");
			}
			const char *title = n->Attribute("title");
			if (title == NULL) {
				throw XMLException("Cannot get attribute 'title'");
			}
			(*V)[id] = title;
		}

		::size_t N = V->size();
		// iterate every link in the graph, and store those links into Set<Edge> object.
		EdgeSet *E = new EdgeSet();

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
			int a;
			int b;
			if (n->QueryIntAttribute("source", &a) != XML_NO_ERROR) {
				throw XMLException("Cannot get int attribute 'source'");
			}
			if (n->QueryIntAttribute("target", &b) != XML_NO_ERROR) {
				throw XMLException("Cannot get int attribute 'target'");
			}
			E->insert(Edge(b, a));
		}

		return new mcmc::Data((void *)V, E, N);
	}
};

}	// namespace preprocess
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_NETSCIENCE_H__
