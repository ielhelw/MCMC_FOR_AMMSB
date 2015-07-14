#include "mcmc/preprocess/netscience.h"
namespace tinyxml2 {

XMLException::XMLException(const std::string &reason) throw()
    : reason("XMLException" + reason) {}

XMLException::~XMLException() throw() {}

const char *XMLException::what() const throw() { return reason.c_str(); }

}  // namespace tinyxml2

namespace mcmc {
namespace preprocess {

using namespace tinyxml2;

typedef std::unordered_map<Vertex, std::string> VertexAttrib;

NetScience::NetScience(const std::string &filename)
    : DataSet(filename == "" ? "datasets/netscience.xml" : filename) {}

NetScience::~NetScience() {}

const mcmc::Data *NetScience::process() {
  // V stores the mapping between node ID and attribute. i.e title, name. etc
  // i.e {0: "WU, C", 1 :CHUA, L"}
  VertexAttrib *V = new VertexAttrib();
  XMLDocument tree;
  if (tree.LoadFile(filename_.c_str()) != XML_NO_ERROR) {
    throw mcmc::IOException("Cannot open " + filename_);
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
                                  <network id="agent x agent"
   isDirected="false">
                                          <link source="1" target="0"
   type="double" value="2.5">
                                          <link source="3" target="2"
   type="double" value="0.x25">
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
  for (XMLElement *n = c->FirstChildElement("node"); n != NULL;
       n = n->NextSiblingElement("node")) {
    Vertex id;
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
  // iterate every link in the graph, and store those links into Set<Edge>
  // object.
  NetworkGraph *E = new NetworkGraph();

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
  for (XMLElement *n = c->FirstChildElement("link"); n != NULL;
       n = n->NextSiblingElement("link")) {
    Vertex a;
    Vertex b;
    if (n->QueryIntAttribute("source", &a) != XML_NO_ERROR) {
      throw XMLException("Cannot get int attribute 'source'");
    }
    if (n->QueryIntAttribute("target", &b) != XML_NO_ERROR) {
      throw XMLException("Cannot get int attribute 'target'");
    }
    Edge e(b, a);
    e.insertMe(E);
  }

  return new mcmc::Data((void *)V, E, N);
}

}  // namespace preprocess
}  // namespace mcmc
