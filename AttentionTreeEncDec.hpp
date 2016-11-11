#pragma once

#include "LSTM.hpp"
#include "TreeLSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include "BlackOut.hpp"
#include <boost/property_tree/xml_parser.hpp> // XML Parser
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "Utils.hpp"
#include "Utils_tempra28.hpp"

class AttentionTreeEncDec{
public:
  enum AttentionType{ DOT,GENERAL,};
  enum ParsedTree{ PARSED, };
  enum InitDec{TREELSTM, CONCAT,};

  class StateNode; //for Tree LSTM's states
  class IndexNode; //for Token Intdices 
  class Data;
  class State;
  class Grad;
  class DecCandidate;
  class ThreadArg;

  AttentionTreeEncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, 
		      std::vector<AttentionTreeEncDec::Data*>& trainData_, std::vector<AttentionTreeEncDec::Data*>& devData_, 
		      const int inputDim, const int hiddenDim, const Real scale,
		      const bool useBlackOut_=false, 
		      const int blackOutSampleNum = 200, const Real blackOutAlpha = 0.4,
		      AttentionTreeEncDec::AttentionType attenType_=AttentionTreeEncDec::DOT, 
		      AttentionTreeEncDec::ParsedTree parsedTree_=AttentionTreeEncDec::PARSED, 
		      AttentionTreeEncDec::InitDec initDec_=AttentionTreeEncDec::TREELSTM,
		      const bool reversed=false, const bool biasOne=false, const Real clipThreshold=3.0,
		      const int beamSize=20, const int maxGeneNum=100,
		      const int miniBatchSize=20, const int threadNum=8, 
		      const Real learningRate=0.5,
		      const bool learningRateShedule_=true,
		      const int vocaThreshold_=1,
		      const bool inputFeeding_=true,
		      const std::string& saveDirName = "pathToSaveDir"); // Set a path to a directory to save a model
  
  AttentionTreeEncDec::AttentionType attenType;
  AttentionTreeEncDec::ParsedTree parsedTree;
  AttentionTreeEncDec::InitDec initDec;

  bool useBlackOut;
  bool reversed, biasOne;
  Real clipThreshold;
  Rand rnd;
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  std::vector<AttentionTreeEncDec::Data*>& trainData;
  std::vector<AttentionTreeEncDec::Data*>& devData;
  LSTM enc, dec;
  TreeLSTM encTree, initDecTree;
  SoftMax softmax;
  BlackOut blackOut;
  MatD sourceEmbed;
  MatD targetEmbed;
  VecD zeros;
  
  MatD Wst, Wct; // Wct; Attention to Tree and Sequence
  VecD bs;
  MatD WgeneralTree; // attenType = AttentionTreeEncDec::GENERAL
  MatD WcellSeq, WcellTree, WhSeq, WhTree; // InitDec = AttentionTreeEncDec::TREELSTM

  int beamSize, maxGeneNum;
  int miniBatchSize, threadNum;
  Real learningRate;
  bool learningRateSchedule;
  int vocaThreshold;
  bool inputFeeding;
  std::string saveDirName;
  int point_counter; // pointer counter for debugging
  int leafCounter;

  AttentionTreeEncDec::StateNode* initTreeEncState;

  void encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState);
  void encodeTree(AttentionTreeEncDec::StateNode* node);
  void encoder(const std::vector<int>& src, AttentionTreeEncDec::State* state);
  void decoder(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state, 
	       VecD& s_tilde, VecD& c0, VecD& s0, const int i);
  void decoderAttention(AttentionTreeEncDec::State* state, std::vector<LSTM::State*>& decState,
			VecD& contextTree, VecD& alphaTree, VecD& s_tilde, const int i);
  void candidateDecoder(AttentionTreeEncDec::State* state, std::vector<LSTM::State*>& decState,
			VecD& s_tilde, const std::vector<int>& tgt, VecD& c0, VecD& s0, const int i);
  void readStat(std::unordered_map<int, std::unordered_map<int, Real> >& stat);
  void translate(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
		 AttentionTreeEncDec::State* state,
		 const int beamSize, const int maxLength, const int showNum);
  void translate(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
		 std::vector<int>& trans, AttentionTreeEncDec::State* state,
		 const int beamSize, const int maxLength);
  void translate2(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
		  std::vector<int>& trans, AttentionTreeEncDec::State* state,
		  const std::unordered_map<int, std::unordered_map<int, Real> >& stat,
		  const int beamSize, const int maxLength);
  void train(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, Real& los, 
	     AttentionTreeEncDec::State* state, std::vector<BlackOut::State*>& blackOutState);
  void showTopAlphaTree(MatD& showAlphaTree, 
			AttentionTreeEncDec::State* state,
			const std::vector<int>& src, const std::vector<int>& tgt);
  void train(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, Real& los, 
	     AttentionTreeEncDec::State* state, std::vector<BlackOut::State*>& blackOutState,
	     std::vector<VecD>& s_tilde, std::vector<VecD>& del_stilde,
	     std::vector<VecD>& contextTreeList);
  void sgd(const AttentionTreeEncDec::Grad& grad, const Real learningRate);
  void backpropThroughStructure(AttentionTreeEncDec::StateNode* parent, 
				AttentionTreeEncDec::StateNode* left, AttentionTreeEncDec::StateNode* right, 
				AttentionTreeEncDec::Grad& grad, AttentionTreeEncDec::State* state);
  void clearTreeNodeState(AttentionTreeEncDec::StateNode* node);
  void train();
  void trainOpenMP();
  void calculateAlpha(const AttentionTreeEncDec::State* state, 
		      const LSTM::State* decState, VecD& alphaTree);
  void calculateAlpha(const AttentionTreeEncDec::State* state, 
		      const LSTM::State* decState, MatD& alphaTree, const int colNum);
  void showTreeNode(const AttentionTreeEncDec::StateNode* node);
  void showTreeNode(const AttentionTreeEncDec::StateNode* node, std::ofstream& outputFile);
  void makeTreeState(const AttentionTreeEncDec::IndexNode* node, 
		     AttentionTreeEncDec::StateNode* stateNode, 
		     AttentionTreeEncDec::State* state);
  void deleteTreeState(AttentionTreeEncDec::StateNode* stateNode);
  void clearTreeState(AttentionTreeEncDec::StateNode* stateNode);
  void makeSubTree(const boost::property_tree::ptree& pt, AttentionTreeEncDec::IndexNode* node, 
		   std::vector<int>& src);
  void makeTree(const std::string& srcParsed, AttentionTreeEncDec::Data* data);
  Real calcLoss(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state);
  Real calcPerplexity(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state);
  Real calcLoss(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state,
		std::vector<BlackOut::State*>& blackOutState);
  void gradChecker(AttentionTreeEncDec::Data* data, MatD& param, const MatD& grad,
		   AttentionTreeEncDec::State* state, 
		   std::vector<BlackOut::State*>& blackOutState);
  void gradChecker(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, 
		   AttentionTreeEncDec::State* state,
		   std::vector<BlackOut::State*>& blackOutState);
  void makeState(const AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state);
  void deleteState(AttentionTreeEncDec::State* state);
  void deleteState(std::vector<BlackOut::State*>& blackOutState);
  void deleteStateList(std::vector<LSTM::State*>& stateList);
  void clearState(AttentionTreeEncDec::State* state, const bool srcParsedFlag);
  void makeTrans(std::vector<int>& tgt, std::vector<int>& trans);
  void loadCorpus(const std::string& src, const std::string& tgt, const std::string& srcParsed, 
		  std::vector<AttentionTreeEncDec::Data*>& data);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
  void saveModel(const float i);
  void saveResult(const Real value, const std::string& name);
  static void demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcParsedTrain, 
		   const std::string& srcDev, const std::string& tgtDev, const std::string& srcParsedDev,
		   const int inputDim, const int hiddenDim, const Real scale,
		   const bool useBlackOut, const int blackOutSampleNum, const Real blackOutAlpha, 
		   const bool reversed, const bool biasOne, const Real clipThreshold,
		   const int beamSize, const int maxGeneNum, 
		   const int miniBatchSize, const int threadNum, 
		   const Real learningRate, const bool learningRateSchedule,
		   const int srcVocaThreshold, const int tgtVocaThreshold,
		   const bool inputFeeding,
		   const std::string& saveDirName);
  static void evaluate(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcParsedTrain, 
		       const std::string& srcDev, const std::string& tgtDev, const std::string& srcParsedDev,
		       const int inputDim, const int hiddenDim, const Real scale,
		       const bool useBlackOut, const int blackOutSampleNum, const Real blackOutAlpha, 
		       const bool reversed,
		       const int beamSize, const int maxGeneNum, 
		       const int miniBatchSize, const int threadNum, 
		       const Real learningRate, const bool learningRateSchedule,
		       const int srcVocaThreshold, const int tgtVocaThreshold,
		       const bool inputFeeding,
		       const std::string& saveDirName, const std::string& loadModelName, const int startIter);
};

class AttentionTreeEncDec::StateNode{
public:
  StateNode(const int tokenIndex_, LSTM::State* state_, StateNode* left_, StateNode* right_):
    tokenIndex(tokenIndex_), state(state_), left(left_), right(right_)
  {}
  
  int tokenIndex;
  LSTM::State* state;
  AttentionTreeEncDec::StateNode* left;
  AttentionTreeEncDec::StateNode* right;
};

class AttentionTreeEncDec::IndexNode{
public:
  IndexNode(const int tokenIndex_, IndexNode* left_, IndexNode* right_):
    tokenIndex(tokenIndex_), left(left_), right(right_)
  {}
  
  int tokenIndex;
  AttentionTreeEncDec::IndexNode* left;
  AttentionTreeEncDec::IndexNode* right;
};

class AttentionTreeEncDec::Data{
public:
  std::vector<int> src, tgt;
  std::vector<int> trans; // Output of Decoder
  AttentionTreeEncDec::IndexNode* srcParsed;
  bool srcParsedFlag;
  AttentionTreeEncDec::State* state;
};

class AttentionTreeEncDec::State{
public:
  std::vector<LSTM::State*> encState, decState;
  AttentionTreeEncDec::StateNode* encTreeState; //rootNode;
  std::vector<AttentionTreeEncDec::StateNode*> encTreeNodeVec; // Tree (Phrase nodes)
  std::vector<AttentionTreeEncDec::StateNode*> encTreeLeafVec; // Tree (Sequence; leaf nodes)
  int leafCounter;
};

class AttentionTreeEncDec::Grad{
public:
  std::unordered_map<int, VecD> sourceEmbed, targetEmbed;
  LSTM::Grad lstmSrcGrad, lstmTgtGrad;
  TreeLSTM::Grad treeLstmEncGrad, treeLstmInitDecGrad;
  SoftMax::Grad softmaxGrad;
  BlackOut::Grad blackOutGrad;
  BlackOut::State blackOutState;
  MatD Wst, Wct; // s~, Wct; Attention to Tree Structure
  VecD bs;
  MatD WgeneralTree; // attenType = AttentionTreeEncDec::GENERAL
  MatD WcellSeq, WcellTree, WhSeq, WhTree; // InitDec = AttentionTreeEncDec::TREELSTM

 
  void init(){
    this->sourceEmbed.clear();
    this->targetEmbed.clear();
    this->lstmSrcGrad.init();
    this->lstmTgtGrad.init();
    this->treeLstmEncGrad.init();
    this->treeLstmInitDecGrad.init();
    this->softmaxGrad.init();
    this->blackOutGrad.init();

    this->Wst.setZero();
    this->bs.setZero();
    this->Wct.setZero();
    this->WgeneralTree.setZero();
    this->WcellSeq.setZero();
    this->WcellTree.setZero();
    this->WhSeq.setZero();
    this->WhTree.setZero();
  }

  Real norm(){
    Real res = 
      this->lstmSrcGrad.norm()
      + this->lstmTgtGrad.norm()
      + this->treeLstmEncGrad.norm()
      + this->treeLstmInitDecGrad.norm()
      + this->softmaxGrad.norm()
      + this->blackOutGrad.norm()
      + this->Wst.squaredNorm()
      + this->bs.squaredNorm()
      + this->Wct.squaredNorm()
      + this->WgeneralTree.squaredNorm()
      + this->WcellSeq.squaredNorm()
      + this->WcellTree.squaredNorm()
      + this->WhSeq.squaredNorm()
      + this->WhTree.squaredNorm();

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }

    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }

    return res;
  }
  
  void operator += (const AttentionTreeEncDec::Grad& grad) {
    this->lstmSrcGrad += grad.lstmSrcGrad;
    this->lstmTgtGrad += grad.lstmTgtGrad;
    this->treeLstmEncGrad += grad.treeLstmEncGrad;  
    this->treeLstmInitDecGrad += grad.treeLstmInitDecGrad;  
    this->softmaxGrad += grad.softmaxGrad;  
    this->blackOutGrad += grad.blackOutGrad;  
    this->Wst += grad.Wst; this->bs += grad.bs; this->Wct += grad.Wct;
    this->WgeneralTree += grad.WgeneralTree;
    this->WcellSeq += grad.WcellSeq; this->WcellTree += grad.WcellTree; 
    this->WhSeq += grad.WhSeq; this->WhTree += grad.WhTree; 

    for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
      if (this->sourceEmbed.count(it->first)){
	this->sourceEmbed.at(it->first) += it->second;
      }
      else {
	this->sourceEmbed[it->first] = it->second;
      }
    }
    for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
      if (this->targetEmbed.count(it->first)){
	this->targetEmbed.at(it->first) += it->second;
      }
      else {
	this->targetEmbed[it->first] = it->second;
      }
    }
  }
  /*
  void operator /= (const Real val){
    this->lstmSrcGrad /= val;
    this->lstmTgtGrad /= val;
    this->treeLstmEncGrad /= val;
    this->treeLstmInitDecGrad /= val;
    this->softmaxGrad /= val;
    this->blackOutGrad /= val;
    this->Wst /= val; this->bs /= val; this->Wct /= val;
    this->WgeneralTree /= val;
    this->WcellSeq /= val; this->WcellTree /= val; 
    this->WhSeq /= val; this->WhTree /= val;     

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      it->second /= val;
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      it->second /= val;
    }
  }
  */
};

class AttentionTreeEncDec::DecCandidate{
public:
  DecCandidate():
    score(0.0), stop(false) 
  {}

  Real score;
  std::vector<int> tgt;
  std::vector<LSTM::State*> decState;
  VecD s_tilde;
  MatD showAlphaTree;
  bool stop;
};

class AttentionTreeEncDec::ThreadArg{
public:
  ThreadArg(AttentionTreeEncDec& attentionTreeEncDec_):
    attentionTreeEncDec(attentionTreeEncDec_), loss(0.0)
  {
    this->grad.lstmSrcGrad = LSTM::Grad(this->attentionTreeEncDec.enc);
    this->grad.lstmTgtGrad = LSTM::Grad(this->attentionTreeEncDec.dec);
    this->grad.treeLstmEncGrad = TreeLSTM::Grad(this->attentionTreeEncDec.encTree);
    this->grad.treeLstmInitDecGrad = TreeLSTM::Grad(this->attentionTreeEncDec.initDecTree);
    if (!this->attentionTreeEncDec.useBlackOut) {
      this->grad.softmaxGrad = SoftMax::Grad(this->attentionTreeEncDec.softmax);
    }
    else{
      this->grad.blackOutState = BlackOut::State(this->attentionTreeEncDec.blackOut);
    }
    this->grad.Wst = MatD::Zero(this->attentionTreeEncDec.Wst.rows(), this->attentionTreeEncDec.Wst.cols());
    this->grad.bs = VecD::Zero(this->attentionTreeEncDec.bs.size());
    this->grad.Wct = MatD::Zero(this->attentionTreeEncDec.Wct.rows(), this->attentionTreeEncDec.Wct.cols());
    this->grad.WgeneralTree = MatD::Zero(this->attentionTreeEncDec.WgeneralTree.rows(), this->attentionTreeEncDec.WgeneralTree.cols());
    this->grad.WcellSeq = MatD::Zero(this->attentionTreeEncDec.WcellSeq.rows(), this->attentionTreeEncDec.WcellSeq.cols());
    this->grad.WcellTree = MatD::Zero(this->attentionTreeEncDec.WcellTree.rows(), this->attentionTreeEncDec.WcellTree.cols());
    this->grad.WhSeq = MatD::Zero(this->attentionTreeEncDec.WhSeq.rows(), this->attentionTreeEncDec.WhSeq.cols());
    this->grad.WhTree = MatD::Zero(this->attentionTreeEncDec.WhTree.rows(), this->attentionTreeEncDec.WhTree.cols());
    for (int i = 0; i< 100; ++i) {
      this->blackOutState.push_back(new BlackOut::State);
    }
    if (this->attentionTreeEncDec.inputFeeding) {
      for (int i = 0; i< 100; ++i) {
	this->s_tilde.push_back(VecD());
	this->del_stilde.push_back(VecD());
	this->contextTreeList.push_back(VecD());
      }
    }
  };

  int beg, end;
  AttentionTreeEncDec& attentionTreeEncDec;
  std::vector<AttentionTreeEncDec::State*> state;
  std::vector<BlackOut::State*> blackOutState;
  std::vector<VecD> s_tilde, del_stilde; // decoder and its gradient for input-feeding
  std::vector<VecD> contextTreeList;
  AttentionTreeEncDec::Grad grad;
  Real loss;
};
