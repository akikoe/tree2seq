#pragma once

#include "LSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include "BlackOut.hpp"

class AttentionEncDec{
public:
  enum AttentionType{ DOT,GENERAL,};

  class StateNode; //for states
  class Data;
  class State;
  class Grad;
  class DecCandidate;
  class ThreadArg;

  AttentionEncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, 
		  std::vector<AttentionEncDec::Data*>& trainData_, std::vector<AttentionEncDec::Data*>& devData_, 
		  const int inputDim, const int hiddenDim, const Real scale,
		  const bool useBlackOut_=false, 
		  const int blackOutSampleNum = 200, const Real blackOutAlpha = 0.4,
		  AttentionEncDec::AttentionType attenType_=AttentionEncDec::DOT, 
		  const bool reversed=false, const bool biasOne=false, const Real clipThreshold=3.0,
		  const int beamSize=20, const int maxGeneNum=100,
		  const int miniBatchSize=20, const int threadNum=8, 
		  const Real learningRate=0.5,
		  const bool learningRateShedule_=true,
		  const int vocaThreshold_=1,
		  const bool inputFeeding_=true,
		  const std::string& saveDirName = "pathToSaveDir"); // Set a path to a directory to save a model
  
  AttentionEncDec::AttentionType attenType;

  bool useBlackOut;
  bool reversed, biasOne;
  Real clipThreshold;
  Rand rnd;
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  std::vector<AttentionEncDec::Data*>& trainData;
  std::vector<AttentionEncDec::Data*>& devData;
  LSTM enc, dec;
  SoftMax softmax;
  BlackOut blackOut;
  MatD sourceEmbed;
  MatD targetEmbed;
  VecD zeros;
  MatD Wst, Wcs; // Wcs; Attention to Sequence
  VecD bs;

  MatD Wgeneral; // attenType = AttentionEncDec::GENERAL

  int beamSize, maxGeneNum;
  int miniBatchSize, threadNum;
  Real learningRate;
  bool learningRateSchedule;
  int vocaThreshold;
  bool inputFeeding;
  std::string saveDirName;
  int point_counter; // pointer counter for debugging
  int leafCounter;

  void encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState);
  void translate(const std::vector<int>& src, AttentionEncDec::State* state,
		 const int beamSize, const int maxLength, const int showNum);
  void translate(const std::vector<int>& src, std::vector<int>& trans, AttentionEncDec::State* state,
		 const int beamSize, const int maxLength);
  void translate2(const std::vector<int>& src, std::vector<int>& trans, AttentionEncDec::State* state,
		  const int beamSize, const int maxLength);
  void train(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, Real& los, 
	     AttentionEncDec::State* state, std::vector<BlackOut::State*>& blackOutState);
  void train(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, Real& los, 
	     AttentionEncDec::State* state, std::vector<BlackOut::State*>& blackOutState,
	     std::vector<VecD>& s_tilde, std::vector<VecD>& del_stilde,
	     std::vector<VecD>& contextSeqList);
  void sgd(const AttentionEncDec::Grad& grad, const Real learningRate);
  void trainOpenMP();
  void calculateAlpha(const AttentionEncDec::State* state, 
		      const LSTM::State* decState, VecD& alphaSeq);
  Real calcLoss(AttentionEncDec::Data* data, AttentionEncDec::State* state);
  Real calcPerplexity(AttentionEncDec::Data* data, AttentionEncDec::State* state);
  Real calcLoss(AttentionEncDec::Data* data, AttentionEncDec::State* state,
		std::vector<BlackOut::State*>& blackOutState);
  void gradChecker(AttentionEncDec::Data* data, MatD& param, const MatD& grad,
		   AttentionEncDec::State* state, 
		   std::vector<BlackOut::State*>& blackOutState);
  void gradChecker(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, 
		   AttentionEncDec::State* state,
		   std::vector<BlackOut::State*>& blackOutState);
  void makeState(const AttentionEncDec::Data* data, AttentionEncDec::State* state);
  void deleteState(AttentionEncDec::State* state);
  void deleteState(std::vector<BlackOut::State*>& blackOutState);
  void clearState(AttentionEncDec::State* state);
  void loadCorpus(const std::string& src, const std::string& tgt, std::vector<AttentionEncDec::Data*>& data);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
  void saveModel(const int i);
  void saveResult(const Real value, const std::string& name);
  static void demo(const std::string& srcTrain, const std::string& tgtTrain,
		   const std::string& srcDev, const std::string& tgtDev,
		   const int inputDim, const int hiddenDim, const Real scale,
		   const bool useBlackOut, const int blackOutSampleNum, const Real blackOutAlpha, 
		   const bool reversed, const bool biasOne, const Real clipThreshold,
		   const int beamSize, const int maxGeneNum, 
		   const int miniBatchSize, const int threadNum, 
		   const Real learningRate, const bool learningRateSchedule,
		   const int srcVocaThreshold, const int tgtVocaThreshold,
		   const bool inputFeeding,
		   const std::string& saveDirName);
  static void evaluate(const std::string& srcTrain, const std::string& tgtTrain,
		       const std::string& srcDev, const std::string& tgtDev,
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

class AttentionEncDec::Data{
public:
  std::vector<int> src, tgt;
  std::vector<int> trans; // Output of Decoder
  AttentionEncDec::State* state;
};

class AttentionEncDec::State{
public:
  std::vector<LSTM::State*> encState, decState;
};

class AttentionEncDec::Grad{
public:
  std::unordered_map<int, VecD> sourceEmbed, targetEmbed;
  LSTM::Grad lstmSrcGrad, lstmTgtGrad;
  SoftMax::Grad softmaxGrad;
  BlackOut::Grad blackOutGrad;
  BlackOut::State blackOutState;
  MatD Wst, Wcs; // s~, Wcs; Attention to Sequence
  VecD bs;
  MatD Wgeneral; // attenType = AttentionEncDec::GENERAL
 
  void init(){
    this->sourceEmbed.clear();
    this->targetEmbed.clear();
    this->lstmSrcGrad.init();
    this->lstmTgtGrad.init();
    this->softmaxGrad.init();
    this->blackOutGrad.init();

    this->Wst.setZero();
    this->bs.setZero();
    this->Wcs.setZero();
    this->Wgeneral.setZero();
  }

  Real norm(){
    Real res = 
      this->lstmSrcGrad.norm()
      + this->lstmTgtGrad.norm()
      + this->softmaxGrad.norm()
      + this->blackOutGrad.norm()
      + this->Wst.squaredNorm()
      + this->bs.squaredNorm()
      + this->Wcs.squaredNorm()
      + this->Wgeneral.squaredNorm();

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }

    return res;
  }
  
  void operator += (const AttentionEncDec::Grad& grad) {
    this->lstmSrcGrad += grad.lstmSrcGrad;
    this->lstmTgtGrad += grad.lstmTgtGrad;
    this->softmaxGrad += grad.softmaxGrad;  
    this->blackOutGrad += grad.blackOutGrad;  
    this->Wst += grad.Wst; this->bs += grad.bs; this->Wcs += grad.Wcs;
    this->Wgeneral += grad.Wgeneral;

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
    this->softmaxGrad /= val;
    this->blackOutGrad /= val;
    this->Wst /= val; this->bs /= val; this->Wcs /= val;
    this->Wgeneral /= val;

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      it->second /= val;
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      it->second /= val;
    }
  }
  */
};

class AttentionEncDec::DecCandidate{
public:
  DecCandidate():
    score(0.0), stop(false)
  {}

  Real score;
  std::vector<int> tgt;
  std::vector<LSTM::State*> decState;
  VecD s_tilde;
  MatD showAlphaSeq;
  bool stop;
};

class AttentionEncDec::ThreadArg{
public:
  ThreadArg(AttentionEncDec& attentionEncDec_):
    attentionEncDec(attentionEncDec_), loss(0.0)
  {
    this->grad.lstmSrcGrad = LSTM::Grad(this->attentionEncDec.enc);
    this->grad.lstmTgtGrad = LSTM::Grad(this->attentionEncDec.dec);
    if (!this->attentionEncDec.useBlackOut) {
      this->grad.softmaxGrad = SoftMax::Grad(this->attentionEncDec.softmax);
    }
    else{
      this->grad.blackOutState = BlackOut::State(this->attentionEncDec.blackOut);
    }
    this->grad.Wst = MatD::Zero(this->attentionEncDec.Wst.rows(), this->attentionEncDec.Wst.cols());
    this->grad.bs = VecD::Zero(this->attentionEncDec.bs.size());
    this->grad.Wcs = MatD::Zero(this->attentionEncDec.Wcs.rows(), this->attentionEncDec.Wcs.cols());
    this->grad.Wgeneral = MatD::Zero(this->attentionEncDec.Wgeneral.rows(), this->attentionEncDec.Wgeneral.cols());

    for (int i = 0; i< 100; ++i) {
      this->blackOutState.push_back(new BlackOut::State);
    }
    if (this->attentionEncDec.inputFeeding) {
      for (int i = 0; i< 100; ++i) {
	this->s_tilde.push_back(VecD());
	this->del_stilde.push_back(VecD());
	this->contextSeqList.push_back(VecD());
      }
    }
  };

  int beg, end;
  AttentionEncDec& attentionEncDec;
  std::vector<AttentionEncDec::State*> state;
  std::vector<BlackOut::State*> blackOutState;
  std::vector<VecD> s_tilde, del_stilde; // decoder and its gradient for input-feeding
  std::vector<VecD> contextSeqList;
  AttentionEncDec::Grad grad;
  Real loss;
};
