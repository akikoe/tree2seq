#include "AttentionEncDec.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include "Utils_tempra28.hpp"
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>

/* Encoder-Decoder (EncDec.cpp) with Attention Mechanism:
   
  1-layer LSTM units with ``Global Attention``.

  Paper: "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al. published in EMNLP2015.
  Pdf: http://arxiv.org/abs/1508.04025

*/

AttentionEncDec::AttentionEncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, 
				 std::vector<AttentionEncDec::Data*>& trainData_, 
				 std::vector<AttentionEncDec::Data*>& devData_, 
				 const int inputDim, const int hiddenDim, const Real scale,
				 const bool useBlackOut_, 
				 const int blackOutSampleNum, const Real blackOutAlpha,
				 AttentionEncDec::AttentionType attenType_, 
				 const bool reversed_, const bool biasOne_, const Real clipThreshold_,
				 const int beamSize_, const int maxGeneNum_,
				 const int miniBatchSize_, const int threadNum_, 
				 const Real learningRate_, const bool learningRateSchedule_,
				 const int vocaThreshold_,
				 const bool inputFeeding_,
				 const std::string& saveDirName_):
attenType(attenType_),useBlackOut(useBlackOut_),
  reversed(reversed_), biasOne(biasOne_), clipThreshold(clipThreshold_),
  sourceVoc(sourceVoc_), targetVoc(targetVoc_),
  trainData(trainData_), devData(devData_),
  beamSize(beamSize_), maxGeneNum(maxGeneNum_), 
  miniBatchSize(miniBatchSize_), threadNum(threadNum_), 
  learningRate(learningRate_), learningRateSchedule(learningRateSchedule_),
  vocaThreshold(vocaThreshold_),
  inputFeeding(inputFeeding_),
  saveDirName(saveDirName_)
{
  this->enc = LSTM(inputDim, hiddenDim); // Encoder; set dimension
  this->enc.init(this->rnd, scale);
  this->dec = LSTM(inputDim, hiddenDim, hiddenDim); // Decoder; set dimension
  this->dec.init(this->rnd, scale);
  if (this->biasOne) { // LSTMs' biases set to 1 
    this->enc.bf.fill(1.0);
    this->dec.bf.fill(1.0);
  }

  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);

  this->zeros = VecD::Zero(hiddenDim); // Zero vector
  // initialize W
  this->Wst = MatD(hiddenDim, hiddenDim); // s_tilde's weight for decoder s_t
  this->Wcs = MatD(hiddenDim, hiddenDim); // .. for Sequnece
  this->bs = VecD::Zero(hiddenDim);       // .. for bias
  this->rnd.uniform(this->Wst, scale);
  this->rnd.uniform(this->Wcs, scale);

  // attentionType == GENERAL
  this->Wgeneral = MatD(hiddenDim, hiddenDim); // attenType == GENERAL; for Sequence
  this->rnd.uniform(this->Wgeneral, scale);

  if (!this->useBlackOut) {
    this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  }
  else {
    VecD freq = VecD(this->targetVoc.tokenList.size());

    for (int i = 0; i < (int)this->targetVoc.tokenList.size(); ++i) {
      freq.coeffRef(i, 0) = this->targetVoc.tokenList[i]->count;
    }
    this->blackOut = BlackOut(hiddenDim, this->targetVoc.tokenList.size(), blackOutSampleNum);
    this->blackOut.initSampling(freq, blackOutAlpha);
  }
}

void AttentionEncDec::encode(const std::vector<int>& src, 
			     std::vector<LSTM::State*>& encState){ // Encoder for sequence
  encState[0]->c = this->zeros;
  encState[0]->h = this->zeros;
  encState[0]->delc = this->zeros;
  encState[0]->delh = this->zeros;
  for (int i = 0; i < (int)src.size(); ++i){
    this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
    encState[i+1]->delc = this->zeros; // (!) Initialize here for backward
    encState[i+1]->delh = this->zeros;
  }
}

struct sort_pred {
  bool operator()(const AttentionEncDec::DecCandidate left, const AttentionEncDec::DecCandidate right) {
    return left.score > right.score;
  }
};

void AttentionEncDec::translate(const std::vector<int>& src,
				AttentionEncDec::State* state,
				const int beamSize, const int maxLength, const int showNum) {
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist; 
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)
  VecD s_tilde = this->zeros; // Decoder; s~

  this->encode(src, state->encState); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      }
      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      if (i == 0) { // initialize decoder's initial state
	candidate[j].decState[i]->c = state->encState.back()->c;
	candidate[j].decState[i]->h = state->encState.back()->h;
      }
      else { // i >= 1
	if (this->inputFeeding) {
	  // input-feeding approach [Luong et al., EMNLP2015]
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].s_tilde, 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
	} else {
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, prev, cur)
	}
      }

      /* Attention */
      candidate[j].s_tilde = this->Wst*candidate[j].decState[i]->h + this->bs; // Set ``s_tilde``
      // sequence
      contextSeq = this->zeros;
      this->calculateAlpha(state, candidate[j].decState[i], alphaSeq);
      candidate[j].showAlphaSeq.row(i) = alphaSeq.transpose();
      for (int k = 1; k < (int)state->encState.size(); ++k) {
	contextSeq += alphaSeq.coeff(k-1, 0)*state->encState[k]->h;
      }
      candidate[j].s_tilde += this->Wcs*contextSeq;
      ActFunc::tanh(candidate[j].s_tilde); // s~tanh(W_c[ht; c_Seq])

      if (!this->useBlackOut) {
	this->softmax.calcDist(candidate[j].s_tilde, targetDist);
      }
      else {
	this->blackOut.calcDist(candidate[j].s_tilde, targetDist);
      }
      score.col(j).array() = candidate[j].score + targetDist.array().log();
    }

    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy;
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].tgt.push_back(row);
      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex) {
      break;
    }
  }

  for (auto it = src.begin(); it != src.end(); ++it) {
    std::cout << this->sourceVoc.tokenList[*it]->str << " "; 
  }
  std::cout << std::endl;

  for (int i = 0; i < showNum; ++i) {
    std::cout << i+1 << " (" << candidate[i].score << "): ";
    for (auto it = candidate[i].tgt.begin(); it != candidate[i].tgt.end(); ++it) {
      std::cout << this->targetVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  }

  for (auto it = src.begin(); it != src.end(); ++it) {
    std::cout << this->sourceVoc.tokenList[*it]->str << " ";
  }
  std::cout << std::endl;

  for (auto it = tgt.begin(); it != tgt.end(); ++it) {
    std::cout << this->targetVoc.tokenList[*it]->str << " ";
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(stateList);

  std::cout << std::endl;
  this->clearState(state);
}

void AttentionEncDec::translate(const std::vector<int>& src,
				std::vector<int>& trans, AttentionEncDec::State* state,
				const int beamSize, const int maxLength) {
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist;
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)

  this->encode(src, state->encState); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      }
      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      if (i == 0) { // initialize decoder's initial state
	candidate[j].decState[i]->c = state->encState.back()->c;
	candidate[j].decState[i]->h = state->encState.back()->h;
      }
      else { // i >= 1
	if (this->inputFeeding) {
	  // input-feeding approach [Luong et al., EMNLP2015]
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].s_tilde, 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
	} else {
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, prev, cur)
	}
      }

      /* Attention */
      candidate[j].s_tilde = this->Wst*candidate[j].decState[i]->h + this->bs; // Set ``s_tilde``
      // sequence
      contextSeq = this->zeros;
      this->calculateAlpha(state, candidate[j].decState[i], alphaSeq);
      candidate[j].showAlphaSeq.row(i) = alphaSeq.transpose();
      for (int k = 1; k < (int)state->encState.size(); ++k) {
	contextSeq += alphaSeq.coeff(k-1, 0)*state->encState[k]->h;
      }
      candidate[j].s_tilde += this->Wcs*contextSeq;
      ActFunc::tanh(candidate[j].s_tilde); // s~tanh(W_c[ht; c_Seq])

      if (!this->useBlackOut) {
	this->softmax.calcDist(candidate[j].s_tilde, targetDist);
      }
      else {
	this->blackOut.calcDist(candidate[j].s_tilde, targetDist);
      }
      score.col(j).array() = candidate[j].score + targetDist.array().log();
    }

    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col);
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].tgt.push_back(row);
      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex) {
      break;
    }
  }

  for (auto it = candidate[0].tgt.begin(); it != candidate[0].tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(stateList);

  this->clearState(state);
}

void AttentionEncDec::translate2(const std::vector<int>& src,
				 std::vector<int>& trans, AttentionEncDec::State* state,
				 const int beamSize, const int maxLength) {
  const Real minScore = -1.0e+05;
  const int srcLen = src.size()-1;
  static std::unordered_map<int, std::unordered_map<int, Real> > stat; // <src, <trg, Real>>; Real = p(len(trg) | len(src))
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist;
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)

  if (stat.empty()){
    std::ifstream ifs("stat.txt");
    std::vector<std::string> res;
    VecD prob;
    int len = 0;

    for (std::string line; std::getline(ifs, line); ++len){
      Utils::split(line, res);
      prob = VecD(res.size());

      for (int i = 0; i < (int)res.size(); ++i){
	prob.coeffRef(i, 0) = atof(res[i].c_str());
      }

      if (prob.sum() == 0.0){
	continue;
      }

      for (int i = 0; i < prob.rows(); ++i){
	if (prob.coeff(i, 0) == 0.0){
	  continue;
	}

	stat[len][i] = prob.coeff(i, 0);
      }
    }

    return;
  }

  this->encode(src, state->encState); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      }
      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      if (i == 0) { // initialize decoder's initial state
	candidate[j].decState[i]->c = state->encState.back()->c;
	candidate[j].decState[i]->h = state->encState.back()->h;
      }
      else { // i >= 1
	if (this->inputFeeding) {
	  // input-feeding approach [Luong et al., EMNLP2015]
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].s_tilde, 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
	} else {
	  this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), 
			    candidate[j].decState[i-1], candidate[j].decState[i]); // (xt, prev, cur)
	}
      }

      /* Attention */
      candidate[j].s_tilde = this->Wst*candidate[j].decState[i]->h + this->bs; // Set ``s_tilde``
      // sequence
      contextSeq = this->zeros;
      this->calculateAlpha(state, candidate[j].decState[i], alphaSeq);
      candidate[j].showAlphaSeq.row(i) = alphaSeq.transpose();
      for (int k = 1; k < (int)state->encState.size(); ++k) {
	contextSeq += alphaSeq.coeff(k-1, 0)*state->encState[k]->h;
      }
      candidate[j].s_tilde += this->Wcs*contextSeq;
      ActFunc::tanh(candidate[j].s_tilde); // s~tanh(W_c[ht; c_Seq])

      if (!this->useBlackOut) {
	this->softmax.calcDist(candidate[j].s_tilde, targetDist);
      }
      else {
	this->blackOut.calcDist(candidate[j].s_tilde, targetDist);
      }
      score.col(j).array() = candidate[j].score + targetDist.array().log();
    }

    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy;
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      if (row == this->targetVoc.eosIndex) {
	if (stat.count(srcLen)) {
	  const int tgtLen = candidateTmp[j].tgt.size();
	      
	  if (stat.at(srcLen).count(tgtLen)){
	    candidateTmp[j].score += log(stat.at(srcLen).at(tgtLen));
	  }
	  else {
	    candidateTmp[j].score = minScore;
	    score.coeffRef(row, col) = minScore;
	    --j;
	    continue;
	  }
	}
	candidateTmp[j].stop = true;
      }
      candidateTmp[j].tgt.push_back(row);
      if (i == 0) {
	score.row(row).fill(minScore);
      } else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex) {
      break;
    }
  }

  for (auto it = candidate[0].tgt.begin(); it != candidate[0].tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(stateList);

  this->clearState(state);
}

void AttentionEncDec::train(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, Real& loss, 
			    AttentionEncDec::State* state, 
			    std::vector<BlackOut::State*>& blackOutState) {
  VecD targetDist;
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  MatD showAlphaSeq = MatD::Zero(data->tgt.size(), state->encState.size()-1);
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~

  VecD del_stilde;
  VecD del_contextSeq = this->zeros;
  VecD del_alphaSeq = VecD::Zero(state->encState.size()-1);
  VecD del_alignScore = VecD::Zero(state->encState.size()-1); // delta for alignment score
  VecD del_c0 = this->zeros, del_s0 = this->zeros;
  loss = 0.0;

  this->encode(data->src, state->encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    if (i == 0) {
      state->decState[i]->c = state->encState.back()->c;
      state->decState[i]->h = state->encState.back()->h;
    } else { // i >= 1 
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]),
			state->decState[i-1], state->decState[i]); // (xt, prev, cur)
    }
    /* Attention */
    s_tilde = this->Wst*state->decState[i]->h + this->bs;
    // sequence
    contextSeq = this->zeros;
    this->calculateAlpha(state, state->decState[i], alphaSeq);
    //showAlphaSeq.row(i) = alphaSeq.transpose();
    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextSeq += alphaSeq.coeff(j-1, 0)*state->encState[j]->h;
    }
    s_tilde += this->Wcs*contextSeq;
    ActFunc::tanh(s_tilde); // s~tanh(W_c[ht; c_Seq])

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
      this->softmax.backward(s_tilde, targetDist, data->tgt[i], del_stilde, grad.softmaxGrad);
    }
    else {
      this->blackOut.sampling(data->tgt[i], grad.blackOutState);
      // *(blackOutState[i]) = grad.blackOutState; // for gradient checker
      this->blackOut.calcSampledDist(s_tilde, targetDist, grad.blackOutState);
      loss += this->blackOut.calcSampledLoss(targetDist);
      this->blackOut.backward(s_tilde, targetDist, grad.blackOutState, del_stilde, grad.blackOutGrad);
    }

    /* Attention's Backpropagation */
    // del_stilde
    del_stilde.array() *= ActFunc::tanhPrime(s_tilde).array();
    state->decState[i]->delh = this->Wst.transpose()*del_stilde;
    grad.Wst += del_stilde*state->decState[i]->h.transpose();
    grad.bs += del_stilde;
    grad.Wcs += del_stilde*contextSeq.transpose();
    del_contextSeq = this->Wcs.transpose()*del_stilde;

    // del_contextSeq
    for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
      state->encState[j+1]->delh += alphaSeq.coeff(j, 0) * del_contextSeq;
      del_alphaSeq.coeffRef(j, 0) = del_contextSeq.dot(state->encState[j+1]->h);
    }
    del_alignScore = alphaSeq.array()*(del_alphaSeq.array()-alphaSeq.dot(del_alphaSeq)); // X.array() - scalar; np.array() -= 1
    if (this->attenType == AttentionEncDec::DOT) { // h^T*s
      for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
	state->encState[j+1]->delh += del_alignScore.coeff(j, 0)*state->decState[i]->h;
	state->decState[i]->delh += del_alignScore.coeff(j, 0)*state->encState[j+1]->h;
      }
    } else if (this->attenType == AttentionEncDec::GENERAL) { // s^T*W*h
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh += (this->Wgeneral.transpose()*state->decState[i]->h)*del_alignScore.coeff(j, 0);
	state->decState[i]->delh += (this->Wgeneral*state->encState[j+1]->h)*del_alignScore.coeff(j, 0);
	grad.Wgeneral += del_alignScore.coeff(j, 0)*state->decState[i]->h*state->encState[j+1]->h.transpose();
      }
    } else {}
  }

  state->decState.back()->delc = this->zeros; 
  for (int i = data->tgt.size()-1; i >= 1; --i) {
    state->decState[i-1]->delc = this->zeros;
    this->dec.backward(state->decState[i-1], state->decState[i], 
		       grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));
    if (grad.targetEmbed.count(data->tgt[i-1])) {
      grad.targetEmbed.at(data->tgt[i-1]) += state->decState[i]->delx;
    }
    else {
      grad.targetEmbed[data->tgt[i-1]] = state->decState[i]->delx;
    }
  }
  // Decoder -> Encoder
  state->encState.back()->delc += state->decState[0]->delc;
  state->encState.back()->delh += state->decState[0]->delh;

  for (int i = data->src.size(); i >= 1; --i) {
    this->enc.backward(state->encState[i-1], state->encState[i], grad.lstmSrcGrad, 
		       this->sourceEmbed.col(data->src[i-1]));
    if (grad.sourceEmbed.count(data->src[i-1])) {
      grad.sourceEmbed.at(data->src[i-1]) += state->encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = state->encState[i]->delx;
    } 
  }
}

void AttentionEncDec::train(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, Real& loss, 
			    AttentionEncDec::State* state, 
			    std::vector<BlackOut::State*>& blackOutState,
			    std::vector<VecD>& s_tilde, std::vector<VecD>& del_stilde, 
			    std::vector<VecD>& contextSeqList) {
  VecD targetDist;
  std::vector<VecD> alphaSeq;
  std::vector<VecD> showAlphaSeq;

  VecD del_contextSeq = this->zeros;
  VecD del_alphaSeq = VecD::Zero(state->encState.size()-1);
  VecD del_alignScore = VecD::Zero(state->encState.size()-1); // delta for alignment score
  VecD del_c0 = this->zeros, del_s0 = this->zeros;
  loss = 0.0;

  this->encode(data->src, state->encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    if (i == 0) {
      state->decState[i]->c = state->encState.back()->c;
      state->decState[i]->h = state->encState.back()->h;
    } else { // i >= 1
      // input-feeding approach [Luong et al., EMNLP2015]
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde[i-1], 
			state->decState[i-1], state->decState[i]); // (xt, at, prev, cur)
    }

    /* Attention */
    s_tilde[i] = this->Wst*state->decState[i]->h + this->bs;
    // sequence
    contextSeqList[i] = this->zeros;
    alphaSeq.push_back(VecD(state->encState.size()-1)); // Vector for attentional weight
    showAlphaSeq.push_back(MatD::Zero(data->tgt.size(), 
				      state->encState.size()-1));
    this->calculateAlpha(state, state->decState[i], alphaSeq[i]);

    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextSeqList[i] += alphaSeq[i].coeff(j-1, 0)*state->encState[j]->h;
    }
    s_tilde[i] += this->Wcs*contextSeqList[i];
    ActFunc::tanh(s_tilde[i]); // s~tanh(W_c[ht; c_Seq])

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde[i], targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
      this->softmax.backward(s_tilde[i], targetDist, data->tgt[i], del_stilde[i], grad.softmaxGrad);
    }
    else {
      this->blackOut.sampling(data->tgt[i], grad.blackOutState);
      // *(blackOutState[i]) = grad.blackOutState; // for gradient checker
      this->blackOut.calcSampledDist(s_tilde[i], targetDist, grad.blackOutState);
      loss += this->blackOut.calcSampledLoss(targetDist);
      this->blackOut.backward(s_tilde[i], targetDist, grad.blackOutState, del_stilde[i], grad.blackOutGrad);
    }
  }

  state->decState.back()->delc = this->zeros; 
  state->decState.back()->delh = this->zeros; 
  state->decState.back()->dela = this->zeros; 
  for (int i = (int)data->tgt.size()-1; i >= 0; --i) {
    /* Attention's Backpropagation */
    // del_stilde 
    if (i < (int)data->tgt.size()-1) {
      del_stilde[i] += state->decState[i+1]->dela; // add gradients to the previous del_stilde 
                                                   // by input-feeding [Luong et al., EMNLP2015]
    } else {}
    del_stilde[i].array() *= ActFunc::tanhPrime(s_tilde[i]).array();
    state->decState[i]->delh += this->Wst.transpose()*del_stilde[i];
    grad.Wst += del_stilde[i]*state->decState[i]->h.transpose();
    grad.bs += del_stilde[i];
    grad.Wcs += del_stilde[i]*contextSeqList[i].transpose();
    del_contextSeq = this->Wcs.transpose()*del_stilde[i];
    // del_contextSeq
    for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
      state->encState[j+1]->delh += alphaSeq[i].coeff(j, 0) * del_contextSeq;
      del_alphaSeq.coeffRef(j, 0) = del_contextSeq.dot(state->encState[j+1]->h);
    }
    del_alignScore = alphaSeq[i].array()*(del_alphaSeq.array()-alphaSeq[i].dot(del_alphaSeq)); // X.array() - scalar; np.array() -= 1
    if (this->attenType == AttentionEncDec::DOT) { // h^T*s
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh += del_alignScore.coeff(j, 0)*state->decState[i]->h;
	state->decState[i]->delh += del_alignScore.coeff(j, 0)*state->encState[j+1]->h;
      }
    } else if (this->attenType == AttentionEncDec::GENERAL) { // s^T*W*h
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh += (this->Wgeneral.transpose()*state->decState[i]->h)*del_alignScore.coeff(j, 0);
	state->decState[i]->delh += (this->Wgeneral*state->encState[j+1]->h)*del_alignScore.coeff(j, 0);
	grad.Wgeneral += del_alignScore.coeff(j, 0)*state->decState[i]->h*state->encState[j+1]->h.transpose();
      }
    } else {}
    if (i > 0) {
      state->decState[i-1]->delc = this->zeros;
      state->decState[i-1]->delh = this->zeros;
      state->decState[i-1]->dela = this->zeros;
      this->dec.backward(state->decState[i-1], state->decState[i], 
			 grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), s_tilde[i-1]);
      if (grad.targetEmbed.count(data->tgt[i-1])) {
	grad.targetEmbed.at(data->tgt[i-1]) += state->decState[i]->delx;
      }
      else {
	grad.targetEmbed[data->tgt[i-1]] = state->decState[i]->delx;
      }
    } else {}
  }

  // Decoder -> Encoder
  state->encState.back()->delc += state->decState[0]->delc;
  state->encState.back()->delh += state->decState[0]->delh;

  for (int i = data->src.size(); i >= 1; --i) {
    this->enc.backward(state->encState[i-1], state->encState[i], grad.lstmSrcGrad, 
		       this->sourceEmbed.col(data->src[i-1]));
    if (grad.sourceEmbed.count(data->src[i-1])) {
      grad.sourceEmbed.at(data->src[i-1]) += state->encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = state->encState[i]->delx;
    }
  }
  
  for (auto it = alphaSeq.begin(); it != alphaSeq.end(); ++it) {
    *it = MatD();
  }
  Utils::swap(alphaSeq);
  for (auto it = showAlphaSeq.begin(); it != showAlphaSeq.end(); ++it) {
    *it = MatD();
  }
  Utils::swap(showAlphaSeq);
}


void AttentionEncDec::calculateAlpha(const AttentionEncDec::State* state,
				     const LSTM::State* decState, VecD& alphaSeq) { // calculate attentional weight;
  if (this->attenType  == AttentionEncDec::DOT) { // inner product; h^T*s
    // Sequnce (leaf)
    for (int i = 1; i < (int)state->encState.size(); ++i) { // encState is 1-origin (encState[0] == h0)
      alphaSeq.coeffRef(i-1, 0) = state->encState[i]->h.dot(decState->h); // coeffRef: insertion
    }
  }
  else if (this->attenType == AttentionEncDec::GENERAL) { // s^T*W*h
    for (int i = 1; i < (int)state->encState.size(); ++i) {
      alphaSeq.coeffRef(i-1, 0) = decState->h.dot(this->Wgeneral * state->encState[i]->h);
    }
  } else {}
  // softmax of ``alphaSeq``
  alphaSeq = alphaSeq.array().exp(); // exp() operation for all elements; np.exp(alphaSeq) 
  alphaSeq /= alphaSeq.array().sum(); // alphaSeq.sum()
}

void AttentionEncDec::sgd(const AttentionEncDec::Grad& grad, const Real learningRate) {
  this->Wst -= learningRate * grad.Wst;
  this->bs -= learningRate * grad.bs;
  this->Wcs -= learningRate * grad.Wcs;

  this->Wgeneral -= learningRate * grad.Wgeneral;

  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    this->sourceEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it) {
    this->targetEmbed.col(it->first) -= learningRate * it->second;
  }
}

void AttentionEncDec::trainOpenMP() { // Multi-threading
  static std::vector<AttentionEncDec::ThreadArg*> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static AttentionEncDec::Grad grad;
  Real lossTrain = 0.0, lossDev = 0.0, tgtNum = 0.0;
  Real gradNorm, lr = this->learningRate;
  struct timeval start, end;

  if (args.empty()) {
    for (int i = 0; i < this->threadNum; ++i) {
      args.push_back(new AttentionEncDec::ThreadArg(*this));
    }
    for (int i = 0, step = this->trainData.size()/this->miniBatchSize; i< step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->miniBatchSize-1)));
      // Create pairs of MiniBatch, e.g. [(0,3), (4, 7), ...]
    }
    // The whole gradients
    grad.lstmSrcGrad = LSTM::Grad(this->enc);
    grad.lstmTgtGrad = LSTM::Grad(this->dec);
    grad.softmaxGrad = SoftMax::Grad(this->softmax);

    grad.Wst = MatD::Zero(this->Wst.rows(), this->Wst.cols());
    grad.bs = VecD::Zero(this->bs.size());
    grad.Wcs = MatD::Zero(this->Wcs.rows(), this->Wcs.cols());
    grad.Wgeneral= MatD::Zero(this->Wgeneral.rows(), this->Wgeneral.cols());
  }

  this->rnd.shuffle(this->trainData);
  gettimeofday(&start, 0);

  int count = 0;
  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it) {
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

    std::unordered_map<int, AttentionEncDec::State*> stateList;
    for (int j = it->first; j <= it->second; ++j) {
      stateList[j] = new AttentionEncDec::State;
      this->makeState(this->trainData[j], stateList.at(j));
    } // prepare states before omp parallel

#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic) shared(args, stateList)
    for (int i = it->first; i <= it->second; ++i) {
      int id = omp_get_thread_num();
      Real loss;
      if(this->inputFeeding) {
	this->train(this->trainData[i], args[id]->grad, loss, stateList[i], args[id]->blackOutState, args[id]->s_tilde, args[id]->del_stilde, args[id]->contextSeqList);
      } else {
	this->train(this->trainData[i], args[id]->grad, loss, stateList[i], args[id]->blackOutState);
      }

      /* ..Gradient Checking.. :) */
      // this->gradChecker(this->trainData[i], args[id]->grad, stateList[i], args[id]->blackOutState);

      args[id]->loss += loss;
    }

    for (int id = 0; id < this->threadNum; ++id) {
      grad += args[id]->grad;
      args[id]->grad.init();
      lossTrain += args[id]->loss;
      args[id]->loss = 0.0;
    }

    // Delete States
    for (auto stateIt = stateList.begin(); stateIt != stateList.end(); ++stateIt) {
      this->deleteState(stateIt->second); // Decoder
    }

    gradNorm = sqrt(grad.norm())/this->miniBatchSize;
    Utils::infNan(gradNorm);
    lr = (gradNorm > this->clipThreshold ? this->clipThreshold*this->learningRate/gradNorm : this->learningRate);
    lr /= this->miniBatchSize;

    // Update the gradients by SGD
    this->enc.sgd(grad.lstmSrcGrad, lr);
    this->dec.sgd(grad.lstmTgtGrad, lr);
    if (!this->useBlackOut) {
      this->softmax.sgd(grad.softmaxGrad, lr);
    }
    else {
      this->blackOut.sgd(grad.blackOutGrad, lr);
    }
    this->sgd(grad, lr);

    grad.init();
  }
  std::cout << std::endl;
  gettimeofday(&end, 0);
  std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << "min." << std::endl;
  std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  gettimeofday(&start, 0);
#pragma omp parallel for num_threads(this->threadNum)
  for (int i = 0; i < (int)this->devData.size(); ++i) {
    Real loss = this->calcLoss(this->devData[i], this->devData[i]->state);
#pragma omp critical
    {
      lossDev += loss;
      tgtNum += this->devData[i]->tgt.size();
    }
  }

  gettimeofday(&end, 0);
  std::cout << "Evaluation time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << "min." << std::endl;
  std::cout << "Development Perplexity and Loss (/sentence):  " 
	    << exp(lossDev/tgtNum) << ", "
	    << lossDev/this->devData.size() << std::endl;

  saveResult(lossTrain/this->trainData.size(), ".trainLoss"); // Training Loss
  saveResult(exp(lossDev/tgtNum), ".devPerp");          // Perplexity
  saveResult(lossDev/this->devData.size(), ".devLoss"); // Development Loss
}

Real AttentionEncDec::calcLoss(AttentionEncDec::Data* data, AttentionEncDec::State* state) {
  VecD targetDist;
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  MatD showAlphaSeq = MatD::Zero(data->tgt.size(), state->encState.size()-1);
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real loss = 0.;

  this->encode(data->src, state->encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    if (i == 0) {
      state->decState[i]->c = state->encState.back()->c;
      state->decState[i]->h = state->encState.back()->h;
    }
    else {
      if (this->inputFeeding) {
	// input-feeding approach [Luong et al., EMNLP2015]
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde, 
			  state->decState[i-1], state->decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
      } else {
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), 
			  state->decState[i-1], state->decState[i]); // (xt, prev, cur)
      }
    }
    /* Attention */
    s_tilde = this->Wst*state->decState[i]->h + this->bs; // Set ``s_tilde``
    // sequence
    contextSeq = this->zeros;
    this->calculateAlpha(state, state->decState[i], alphaSeq);
    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextSeq += alphaSeq.coeff(j-1, 0)*state->encState[j]->h;
    }
    s_tilde += this->Wcs*contextSeq;
    ActFunc::tanh(s_tilde); // s~tanh(W_c[ht])

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    }
    else {
      this->blackOut.calcDist(s_tilde, targetDist); //Softmax
      loss += this->blackOut.calcLoss(targetDist, data->tgt[i]); // Softmax
    }
  }
  this->clearState(state);
  return loss;
}

Real AttentionEncDec::calcPerplexity(AttentionEncDec::Data* data, AttentionEncDec::State* state) { // calculate the perplexity without `*EOS*`
  VecD targetDist;
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  MatD showAlphaSeq = MatD::Zero(data->tgt.size(), state->encState.size()-1);
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real perp = 0.; // Perplexity

  this->encode(data->src, state->encState);

  for (int i = 0; i < (int)data->tgt.size()-1; ++i) { // omit `*EOS*` in data->tgt
    if (i == 0) {
      state->decState[i]->c = state->encState.back()->c;
      state->decState[i]->h = state->encState.back()->h;
    }
    else {
      if (this->inputFeeding) {
	// input-feeding approach [Luong et al., EMNLP2015]
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde, 
			  state->decState[i-1], state->decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
      } else {
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), 
			  state->decState[i-1], state->decState[i]); // (xt, prev, cur)
      }
    }
    /* Attention */
    s_tilde = this->Wst*state->decState[i]->h + this->bs; // Set ``s_tilde``
    // sequence
    contextSeq = this->zeros;
    this->calculateAlpha(state, state->decState[i], alphaSeq);
    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextSeq += alphaSeq.coeff(j-1, 0)*state->encState[j]->h;
    }
    s_tilde += this->Wcs*contextSeq;
    ActFunc::tanh(s_tilde); // s~tanh(W_c[ht])

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
    }
    else {
      this->blackOut.calcDist(s_tilde, targetDist); //Softmax
    }
    perp -= log(targetDist.coeff(data->tgt[i], 0)); // Perplexity
  }
  this->clearState(state);
  return exp(perp/(data->tgt.size()-1)); // Perplexity without `*EOS*`
}

Real AttentionEncDec::calcLoss(AttentionEncDec::Data* data, AttentionEncDec::State* state,
			       std::vector<BlackOut::State*>& blackOutState) { // for gradient checker
  VecD targetDist;
  VecD alphaSeq = VecD(state->encState.size()-1); // Vector for attentional weight
  MatD showAlphaSeq = MatD::Zero(data->tgt.size(), state->encState.size()-1);
  VecD contextSeq; // C_Seq = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real loss = 0.;

  this->encode(data->src, state->encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    if (i == 0) {
      state->decState[i]->c = state->encState.back()->c;
      state->decState[i]->h = state->encState.back()->h;
    }
    else {
      if (this->inputFeeding) {
	// input-feeding approach [Luong et al., EMNLP2015]
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde, 
			  state->decState[i-1], state->decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
      } else {
	this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), 
			  state->decState[i-1], state->decState[i]); // (xt, prev, cur)
      }
    }
    /* Attention */
    s_tilde = this->Wst*state->decState[i]->h + this->bs; // Set ``s_tilde``
    // sequence
    contextSeq = this->zeros;
    this->calculateAlpha(state, state->decState[i], alphaSeq);
    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextSeq += alphaSeq.coeff(j-1, 0)*state->encState[j]->h;
    }
    s_tilde += this->Wcs*contextSeq;
    ActFunc::tanh(s_tilde); // s~tanh(W_c[ht])
    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    }
    else {
      this->blackOut.calcSampledDist(s_tilde, targetDist, *(blackOutState[i]));
      loss += this->blackOut.calcSampledLoss(targetDist);
    }
  }
  this->clearState(state);
  return loss;
}

void AttentionEncDec::gradChecker(AttentionEncDec::Data* data, MatD& param, const MatD& grad,
				  AttentionEncDec::State* state,
				  std::vector<BlackOut::State*>& blackOutState) {
  const Real EPS = 1.0e-04;
  Real val= 0.0;
  Real objFuncPlus = 0.0, objFuncMinus = 0.0;

  std::cout << std::endl;
  for (int i = 0; i < param.rows(); ++i) {
    for (int j = 0; j < param.cols(); ++j) {
      val = param.coeff(i, j); // Θ_i
      param.coeffRef(i, j) = val + EPS;
      objFuncPlus = this->calcLoss(data, state, blackOutState);      
      param.coeffRef(i, j) = val - EPS;
      objFuncMinus = this->calcLoss(data, state, blackOutState);
      param.coeffRef(i, j) = val;

      std::cout << "Grad: " << grad.coeff(i, j) << std::endl ;
      std::cout << "Enum: " << (objFuncPlus - objFuncMinus)/(2.0*EPS) << std::endl ;
    }
  }
}

void AttentionEncDec::gradChecker(AttentionEncDec::Data* data, AttentionEncDec::Grad& grad, 
				  AttentionEncDec::State* state, 
				  std::vector<BlackOut::State*>& blackOutState) {
  const Real EPS = 1.0e-04;
  Real val= 0.0;
  Real objFuncPlus = 0.0, objFuncMinus = 0.0;

  std::cout << std::endl;
  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      val = this->sourceEmbed.coeff(i, it->first);
      this->sourceEmbed.coeffRef(i, it->first) = val + EPS;
      objFuncPlus = this->calcLoss(data, state, blackOutState);
      this->sourceEmbed.coeffRef(i, it->first) = val - EPS;
      objFuncMinus = this->calcLoss(data, state, blackOutState);
      this->sourceEmbed.coeffRef(i, it->first) = val;

      std::cout << "Grad: " << it->second.coeff(i, 0) << std::endl ;
      std::cout << "Enum: " << (objFuncPlus - objFuncMinus)/(2.0*EPS) << std::endl ;
    }
  }
}

void AttentionEncDec::makeState(const AttentionEncDec::Data* data, AttentionEncDec::State* state) {
  // Src
  for (auto it = data->src.begin(); it != data->src.end(); ++it) { // includes "End of Sentence" mark
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    state->encState.push_back(lstmState);
  }
  LSTM::State *lstmState(NULL);
  lstmState = new LSTM::State;
  state->encState.push_back(lstmState); // for initial hidden state; h_0
  // Tgt
  for (auto it = data->tgt.begin(); it != data->tgt.end(); ++it) { // includes "End of Sentence" mark
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    state->decState.push_back(lstmState);
  }
}

void AttentionEncDec::deleteState(AttentionEncDec::State* state) {
  // Src
  for (auto it = state->encState.begin(); it != state->encState.end(); ++it) { // includes "End of Sentence" mark
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(state->encState);
  // Tgt
  for (auto it = state->decState.begin(); it != state->decState.end(); ++it) { // includes "End of Sentence" mark
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(state->decState);
}

void AttentionEncDec::deleteState(std::vector<BlackOut::State*>& blackOutState) {
  // BlackOut
  for (auto it = blackOutState.begin(); it != blackOutState.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(blackOutState);
}

void AttentionEncDec::clearState(AttentionEncDec::State* state) {
  // Src
  for (auto it = state->encState.begin(); it != state->encState.end(); ++it) { // includes "End of Sentence" mark
    (*it)->clear();
  }
  // Tgt
  for (auto it = state->decState.begin(); it != state->decState.end(); ++it) { // includes "End of Sentence" mark
    (*it)->clear();
  }
}

void AttentionEncDec::loadCorpus(const std::string& src, const std::string& tgt,
				 std::vector<AttentionEncDec::Data*>& data) {
  std::ifstream ifsSrc(src.c_str());
  std::ifstream ifsTgt(tgt.c_str());

  assert(ifsSrc);
  assert(ifsTgt);

  int numLine = 0;
  // Src
  for (std::string line; std::getline(ifsSrc, line);) {
    std::vector<std::string> tokens;
    data.push_back(new AttentionEncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }
    if (this->reversed)
      std::reverse(data.back()->src.begin(), data.back()->src.end());
    data.back()->src.push_back(sourceVoc.eosIndex); // Append "EndOfSentence" mark
  }

  //Tgt
  for (std::string line; std::getline(ifsTgt, line);) {
    std::vector<std::string> tokens;

    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }
    data[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }
}

void AttentionEncDec::saveModel(const int i) {
  std::ostringstream oss;

  oss << this->saveDirName << "Model_AttentionEncDec"
      << ".itr_" << i+1
      << ".BlackOut_" << (this->useBlackOut?"true":"false")
      << ".reversed_" << (this->reversed?"true":"false")
      << ".beamSize_" << this->beamSize 
      << ".miniBatchSize_" << this->miniBatchSize
      << ".threadNum_" << this->threadNum
      << ".lrSGD_"<< this->learningRate 
      << ".lrSchedule_"<<  (this->learningRateSchedule?"true":"false")
      << ".vocaThreshold_"<< this->vocaThreshold
      << ".inputFeeding_"<< (this->inputFeeding?"true":"false")
      << ".bin"; 
  this->save(oss.str());
}

void AttentionEncDec::saveResult(const Real value, const std::string& name) {
  /* For Model Analysis */
  std::ofstream valueFile;
  std::ostringstream ossValue;
  ossValue << this->saveDirName << "Model_AttentionEncDec" << name;

  valueFile.open(ossValue.str(), std::ios::app); // open a file with 'a' mode

  valueFile << value << std::endl;
}

void AttentionEncDec::demo(const std::string& srcTrain, const std::string& tgtTrain, 
			   const std::string& srcDev, const std::string& tgtDev,
			   const int inputDim, const int hiddenDim, const Real scale,
			   const bool useBlackOut, const int blackOutSampleNum, const Real blackOutAlpha, 
			   const bool reversed, const bool biasOne, const Real clipThreshold,
			   const int beamSize, const int maxGeneNum,
			   const int miniBatchSize, const int threadNum,
			   const Real learningRate, const bool learningRateSchedule,
			   const int srcVocaThreshold, const int tgtVocaThreshold,
			   const bool inputFeeding,
			   const std::string& saveDirName) {
  Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  Vocabulary targetVoc(tgtTrain, tgtVocaThreshold);
  std::vector<AttentionEncDec::Data*> trainData, devData;

  AttentionEncDec attentionEncDec(sourceVoc, targetVoc, trainData, devData, 
				  inputDim, hiddenDim, scale,
				  useBlackOut, blackOutSampleNum, blackOutAlpha,
				  AttentionEncDec::DOT, 
				  reversed, biasOne, clipThreshold,
				  beamSize, maxGeneNum, 
				  miniBatchSize, threadNum,
				  learningRate, learningRateSchedule, 
				  srcVocaThreshold,
				  inputFeeding,
				  saveDirName);

  attentionEncDec.loadCorpus(srcTrain, tgtTrain, trainData);
  attentionEncDec.loadCorpus(srcDev, tgtDev, devData); 
  for (int i = 0; i < (int)devData.size(); ++i) { // make AttentionEncDec::State* for devData
    devData[i]->state = new AttentionEncDec::State;
    attentionEncDec.makeState(devData[i], devData[i]->state);
  }

  auto test = trainData[0];
  test->state = new AttentionEncDec::State;
  attentionEncDec.makeState(test, test->state);
  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  for (int i = 0; i < 100; ++i) {
    if (attentionEncDec.learningRateSchedule && i > 4) attentionEncDec.learningRate *= 0.5; 
    std::cout << "\nEpoch " << i+1 << " (lr = " << attentionEncDec.learningRate << ")" << std::endl;
    attentionEncDec.trainOpenMP();
    // Save a model
    attentionEncDec.saveModel(i);
    std::cout << "** Greedy Search" << std::endl;
    attentionEncDec.translate(test->src, test->state, 1, attentionEncDec.maxGeneNum, 1);
    std::cout << "** Beam Search" << std::endl;
    attentionEncDec.translate(test->src, test->state, attentionEncDec.beamSize, attentionEncDec.maxGeneNum, 5);
  }
}

void AttentionEncDec::evaluate(const std::string& srcTrain, const std::string& tgtTrain,
			       const std::string& srcTest, const std::string& tgtTest,
			       const int inputDim, const int hiddenDim, const Real scale,
			       const bool useBlackOut, const int blackOutSampleNum, const Real blackOutAlpha, 
			       const bool reversed,
			       const int beamSize, const int maxGeneNum,
			       const int miniBatchSize, const int threadNum,
			       const Real learningRate, const bool learningRateSchedule,
			       const int srcVocaThreshold, const int tgtVocaThreshold,
			       const bool inputFeeding,
			       const std::string& saveDirName, const std::string& loadModelName, const int startIter) {
  static Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold);
  static std::vector<AttentionEncDec::Data*> trainData, testData;

  static AttentionEncDec attentionEncDec(sourceVoc, targetVoc, trainData, testData, 
					 inputDim, hiddenDim, scale,
					 useBlackOut, blackOutSampleNum, blackOutAlpha, 
					 AttentionEncDec::DOT, 
					 reversed, false, 3.0,
					 beamSize, maxGeneNum, 
					 miniBatchSize, threadNum,
					 learningRate, learningRateSchedule,
					 srcVocaThreshold,
					 inputFeeding,
					 saveDirName);
  
  if (testData.empty()) {
    attentionEncDec.loadCorpus(srcTest, tgtTest, testData); 
    for (int i = 0; i< (int)testData.size(); ++i) { // make AttentionEncDec::State* for testData
      testData[i]->state = new AttentionEncDec::State;
      attentionEncDec.makeState(testData[i], testData[i]->state);
    }
    std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
    std::cout << "# of Evaluation Data:\t" << testData.size() << std::endl;
    std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
    std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  } else {}

  // Model Loaded...
  attentionEncDec.load(loadModelName);

  Real lossTest = 0., tgtNum = 0.;

#pragma omp parallel for num_threads(attentionEncDec.threadNum) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    Real loss = attentionEncDec.calcLoss(testData[i], testData[i]->state);
#pragma omp critical
    {
      lossTest += loss;
      tgtNum += testData[i]->tgt.size(); // include `*EOS*`
    }
  }

  std::cout << "Evaluation Data Perplexity and Loss (/sentence):  " 
	    << exp(lossTest/tgtNum) << ", "
	    << lossTest/testData.size() << "; "
	    << testData.size() << std::endl;

  attentionEncDec.translate2(testData[0]->src, testData[0]->trans,
			     testData[0]->state, attentionEncDec.beamSize, attentionEncDec.maxGeneNum); // for loading stats

#pragma omp parallel for num_threads(attentionEncDec.threadNum) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    attentionEncDec.translate2(evalData->src, evalData->trans,
			       evalData->state, attentionEncDec.beamSize, attentionEncDec.maxGeneNum);
  }

  std::ofstream outputFile;
  std::ostringstream oss;
  std::string parsedMode;
  oss << attentionEncDec.saveDirName << "Model_AttentionEncDec"
      << ".BlackOut_" << (attentionEncDec.useBlackOut?"true":"false")
      << ".beamSize_" << attentionEncDec.beamSize 
      << ".miniBatchSize_" << attentionEncDec.miniBatchSize
      << ".threadNum_" << attentionEncDec.threadNum
      << ".lrSGD_"<< attentionEncDec.learningRate 
      << ".lrSchedule_"<<  (attentionEncDec.learningRateSchedule?"true":"false")
      << ".vocaThreshold_"<< attentionEncDec.vocaThreshold
      << ".inputFeeding_"<< (attentionEncDec.inputFeeding?"true":"false")
      << ".startIter_"<< startIter
      << ".OutputDev.translate2"; // or OutputTest
  outputFile.open(oss.str(), std::ios::out);

  for (int i=0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    for (auto it = evalData->trans.begin(); it != evalData->trans.end(); ++it) {
      outputFile << attentionEncDec.targetVoc.tokenList[*it]->str << " ";
    }
    outputFile << std::endl;
    // trans
    testData[i]->trans.clear();

  }
}

void AttentionEncDec::save(const std::string& fileName) {
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);  
  
  this->enc.save(ofs);
  this->dec.save(ofs);
  this->softmax.save(ofs);
  this->blackOut.save(ofs);
  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);

  Utils::save(ofs, Wst);
  Utils::save(ofs, bs);
  Utils::save(ofs, Wcs);

  Utils::save(ofs, Wgeneral);
}

void AttentionEncDec::load(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->enc.load(ifs);
  this->dec.load(ifs);
  this->softmax.load(ifs);
  this->blackOut.load(ifs);
  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
  
  Utils::load(ifs, Wst);
  Utils::load(ifs, bs);
  Utils::load(ifs, Wcs);

  Utils::load(ifs, Wgeneral);
}
