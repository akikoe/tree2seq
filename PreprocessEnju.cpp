#include "PreprocessEnju.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <sys/time.h>

#define print(var)  \
  std::cout<<(var)<<std::endl

void PreprocessEnju::readSubTree(const boost::property_tree::ptree& pt, std::string& outputLine) { // Binary Tree
  /* Cons->Cons(R) ->Cons(L); Cons->Cons, Cons->Tok(=Tok) */
  int numNode = 0;
  numNode = pt.size() - pt.count("<xmlattr>");
  if (numNode == 2) { // two child nodes of (cons, cons)
    int right = 0;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &child_, pt.get_child("")) {
      if (boost::lexical_cast<std::string>(child_.first.data()) == "cons") {
	if (right < 1) { // left node
	  this->readSubTree(child_.second, outputLine);
	}
	else { // right node
	  this->readSubTree(child_.second, outputLine);
	}
	++right;
      }
      else{ // xmlattr
      }
    }
  }
  else if (numNode == 1) { // one child node
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &child_, pt.get_child("")) {
      if(boost::lexical_cast<std::string>(child_.first.data()) == "tok") { // integrate two nodes to one node; ((cons) -> (tok)) => (node->tok)
	boost::optional<std::string> tok_ = boost::lexical_cast<std::string>(child_.second.data());
	outputLine.append(" ");
	outputLine.append(tok_.get());
	return;
       }
      else if(boost::lexical_cast<std::string>(child_.first.data()) == "cons") {
	this->readSubTree(child_.second, outputLine);
      }
    }
  }
}


void PreprocessEnju::readTree(const std::string& Parsed, std::string& outputLine, bool& parsedSuccess){
  boost::property_tree::ptree pt;
  std::istringstream input(Parsed);
  outputLine = std::string();

  boost::property_tree::xml_parser::read_xml(input, pt);

  boost::optional<std::string> parseStatus = pt.get_optional<std::string>("sentence.<xmlattr>.parse_status");
  if (parseStatus.get() == "success") { // Successfully parsed
    parsedSuccess = true;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &root_, pt.get_child("sentence")) {
      if (boost::lexical_cast<std::string>(root_.first.data()) == "cons") {
	this->readSubTree(root_.second, outputLine);	
      }
    }
    boost::optional<std::string> pp = pt.get_optional<std::string>("sentence");
    if (pp.get().size() > 0){
      outputLine.append(" ");
      outputLine.append(pp.get());
    }
    ++(this->successNum);
  }
  else if (parseStatus.get() == "fragmental parse"){
    parsedSuccess = false;
    ++(this->fragmentalNum);
  }
  else{
    parsedSuccess = false;
    std::cout << parseStatus.get() << std::endl;
    print("Error!!!");
    exit(1);
  }
  std::cout << "\r"
	    << "success: " << this->successNum << ", fragmental parse:" << this->fragmentalNum << std::flush;
}

void PreprocessEnju::extractParsedSentence(const std::string& srcParsed, const std::string& srcFileName, 
					   const std::string& srcParsedFileName, const std::string& tgt, 
					   const std::string& tgtFileName, const int threshold){
  std::ifstream ifsSrcParsed(srcParsed.c_str());
  std::ifstream ifsTgt(tgt.c_str());
  assert(ifsSrcParsed);
  assert(ifsTgt);
  std::ofstream outputSrcFile, outputSrcParsedFile, outputTgtFile;
  outputSrcFile.open(srcFileName, std::ios::out);
  outputSrcParsedFile.open(srcParsedFileName, std::ios::out);
  outputTgtFile.open(tgtFileName, std::ios::out);

  std::string srcOutputLine, tgtOutputLine;
  std::string tgtLine;
  std::vector<std::string> srcTokens, tgtTokens;
  int numLine = 0, numLineLess = 0;

  int successNumTmp, fragmentalNumTmp;
  bool parsedSuccess=true;
  PreprocessEnju preprocessEnju;

  for (std::string srcLine, tgtLine; std::getline(ifsSrcParsed, srcLine) && std::getline(ifsTgt, tgtLine);){
    srcTokens = std::vector<std::string>();
    tgtTokens = std::vector<std::string>();

    successNumTmp = preprocessEnju.successNum;
    fragmentalNumTmp = preprocessEnju.fragmentalNum;

    preprocessEnju.readTree(srcLine, srcOutputLine, parsedSuccess);
    if (parsedSuccess) { // extract the sentences which is successfully parsed
      Utils::split(srcOutputLine, srcTokens);
      Utils::split(tgtLine, tgtTokens);
      if ((int)srcTokens.size() <= threshold && (int)tgtTokens.size() <= threshold) { // remove a sentence exceeding N (``threshold``) words
	outputSrcFile << srcOutputLine << std::endl;
	outputSrcParsedFile << srcLine << std::endl;
	outputTgtFile << tgtLine << std::endl;
	++numLineLess;
      }
      else{
	--preprocessEnju.successNum;
      }
    }
    else{
      if ( preprocessEnju.fragmentalNum > fragmentalNumTmp){
	--preprocessEnju.fragmentalNum;
      }
    }
    std::cout << ", Progress: " << numLineLess << "/" << ++numLine << " lines" << std::flush;
  }
}
