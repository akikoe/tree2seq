#pragma once

#include <boost/property_tree/xml_parser.hpp> // XML Parser
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "Vocabulary.hpp"

class PreprocessEnju{
public:
  PreprocessEnju():
    successNum(0), fragmentalNum(0)
  {}
  int successNum,fragmentalNum;

  void readSubTree(const boost::property_tree::ptree& pt, std::string& outpuLine);
  void readTree(const std::string& Parsed, std::string& outputLine, bool& parsedSuccess);
  static void extractParsedSentence(const std::string& srcParsed, const std::string& srcFileName, 
				    const std::string& srcParsedFileName, const std::string& tgt, 
				    const std::string& tgtFileName, const int threshold);
};
