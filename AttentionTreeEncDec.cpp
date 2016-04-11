#include "AttentionTreeEncDec.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include "Utils_tempra28.hpp"
#include <sstream>
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>

/* Tree_Encoder-Decoder (TreeEncDec.cpp) with Attention Mechanism:
   
   Tree encoder is constructed following to a binary tree. 
   -- We use English phrase structure parser ``Enju``.
   Each non-leaf node of the tree denotes a phrase, e.g. "green tea",
   and each leaf node does a word, e.g. "tea".
   -- The hidden units of leaf nodes are calculated as the hidden units of sequential Encoder in advance.

  The model pays attention to all non-leaf and leaf nodes.
*/

AttentionTreeEncDec::AttentionTreeEncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, 
					 std::vector<AttentionTreeEncDec::Data*>& trainData_, 
					 std::vector<AttentionTreeEncDec::Data*>& devData_, 
					 const int inputDim, const int hiddenDim, const Real scale,
					 const bool useBlackOut_, 
					 const int blackOutSampleNum, const Real blackOutAlpha,
					 AttentionTreeEncDec::AttentionType attenType_, 
					 AttentionTreeEncDec::ParsedTree parsedTree_, 
					 AttentionTreeEncDec::InitDec initDec_, 
					 const bool reversed_, const bool biasOne_, const Real clipThreshold_,
					 const int beamSize_, const int maxGeneNum_,
					 const int miniBatchSize_, const int threadNum_, 
					 const Real learningRate_, const bool learningRateSchedule_,
					 const int vocaThreshold_,
					 const bool inputFeeding_,
					 const std::string& saveDirName_):
attenType(attenType_), parsedTree(parsedTree_), initDec(initDec_), useBlackOut(useBlackOut_), 
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
  this->encTree = TreeLSTM(inputDim, hiddenDim); // Encoder Tree; set dimension
  this->encTree.init(this->rnd, scale);
  this->initDecTree = TreeLSTM(inputDim, hiddenDim); // Decoder's Initial Tree; set dimension
  this->initDecTree.init(this->rnd, scale);
  if (this->biasOne) { // LSTMs' biases set to 1 
    this->enc.bf.fill(1.0);
    this->dec.bf.fill(1.0);
    this->encTree.bfl.fill(1.0);
    this->encTree.bfr.fill(1.0);
    this->initDecTree.bfl.fill(1.0);
    this->initDecTree.bfr.fill(1.0);
  }
  // 
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);

  this->zeros = VecD::Zero(hiddenDim); // Zero vector
  // initialize W
  this->Wst = MatD(hiddenDim, hiddenDim); // s_tilde's weight for decoder s_t
  this->Wct = MatD(hiddenDim, hiddenDim); // .. for Tree and Sequnece
  this->bs = VecD::Zero(hiddenDim);       // .. for bias
  this->rnd.uniform(this->Wst, scale);
  this->rnd.uniform(this->Wct, scale);

  // attentionType == GENERAL
  this->WgeneralTree = MatD(hiddenDim, hiddenDim); // attenType == GENERAL; for Tree and Sequence
  this->rnd.uniform(this->WgeneralTree, scale);
                                                //       attentional weight set to a uniform distribution 
                                                //       beacuase of score(i, j) = exp(0) = 1
  // initDec == CONCAT
  this->WcellSeq = MatD(hiddenDim, hiddenDim); // Initial Decoder LSTM (cell memory); c = tanh(W0*c_seq + W1*c_tree)
  this->WcellTree = MatD(hiddenDim, hiddenDim);
  this->WhSeq = MatD(hiddenDim, hiddenDim);    // ................... (hidden state); h = tanh(W2*h_seq + W3*h_tree)
  this->WhTree = MatD(hiddenDim, hiddenDim);
  this->rnd.uniform(this->WcellSeq, scale);
  this->rnd.uniform(this->WcellTree, scale);
  this->rnd.uniform(this->WhSeq, scale);
  this->rnd.uniform(this->WhTree, scale);

  // for Tree Encoder
  this->initTreeEncState = 
    new AttentionTreeEncDec::StateNode(-1, new TreeLSTM::State, 0, 0); // tokenIndex=-1; (root or node);
  this->initTreeEncState->state->h = this->zeros;
  this->initTreeEncState->state->c = this->zeros;
  this->initTreeEncState->state->delc = this->zeros;

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

void AttentionTreeEncDec::encode(const std::vector<int>& src, 
				 std::vector<LSTM::State*>& encState) { // Encoder for sequence
  this->enc.forward(this->sourceEmbed.col(src[0]), encState[1]);
  encState[1]->delc = this->zeros; // (!) Initialize here for backward
  encState[1]->delh = this->zeros;
  for (int i = 1; i < (int)src.size(); ++i) {
    this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
    encState[i+1]->delc = this->zeros; // (!) Initialize here for backward
    encState[i+1]->delh = this->zeros;
  }
}

void AttentionTreeEncDec::encodeTree(AttentionTreeEncDec::StateNode* node) {
  node->state->delh = this->zeros;
  node->state->delc = this->zeros;
  // leaf
  if (node->tokenIndex >= 0) {
    return;
  }
  // non-leaf
  else {
    this->encodeTree(node->left);
    this->encodeTree(node->right);
    this->encTree.forward((TreeLSTM::State*)node->state, node->left->state, node->right->state);
  }
}

struct sort_pred {
  bool operator()(const AttentionTreeEncDec::DecCandidate left, const AttentionTreeEncDec::DecCandidate right) {
    return left.score > right.score;
  }
};

void AttentionTreeEncDec::encoder(const std::vector<int>& src, AttentionTreeEncDec::State* state) { // Encoder Part
  this->encode(src, state->encState); // encoder
  if (state->encTreeNodeVec.size() > 0) {
    for (int i = 0; i < (int)state->encTreeLeafVec.size(); ++i) {
      state->encTreeLeafVec[i]->state->c = state->encState[i+1]->c; // ``encState`` is 1-origin; encState[0] == h0
      state->encTreeLeafVec[i]->state->h = state->encState[i+1]->h;
    }
    this->encodeTree(state->encTreeState); // encoder for TREE
    //this->showTreeNode(state->encTreeState);
  } else {}  
}

void AttentionTreeEncDec::decoder(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state,
				  VecD& s_tilde, VecD& c0, VecD& s0, const int i) {

  if (i == 0) { // initialize decoder's initial state
    if (initDec == AttentionTreeEncDec::TREELSTM) { // Left: Sequence, Right: Tree (Phrase node)
      // copy ``c`` and ``h``
      this->initDecTree.forward((TreeLSTM::State*)state->decState[i], 
				state->encState.back(), state->encTreeState->state);
    } else if (initDec == AttentionTreeEncDec::CONCAT) { // inspired by (Bahdanau et al., 2015)
      c0 = this->WcellSeq*state->encState.back()->c
	+ this->WcellTree*state->encTreeState->state->c;
      s0 = this->WhSeq*state->encState.back()->h
	+ this->WhTree*state->encTreeState->state->h;
      ActFunc::tanh(c0);
      ActFunc::tanh(s0);
      state->decState[i]->c = c0; // tanh(W[cell_Seq; cell_Tree])
      state->decState[i]->h = s0; // tanh(W[h_Seq; h_Tree])
    } else {}
  }
  else { // i >= 1
    if (this->inputFeeding) {
      // input-feeding approach [Luong et al., EMNLP2015]
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde, 
			state->decState[i-1], state->decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
    } else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), 
			state->decState[i-1], state->decState[i]); // (xt, prev, cur)
    }
  }
}

void AttentionTreeEncDec::decoderAttention(AttentionTreeEncDec::State* state, std::vector<LSTM::State*>& decState,
					   VecD& contextTree, VecD& alphaTree, VecD& s_tilde, const int i) {
  /* Attention */
  s_tilde = this->bs;
  s_tilde.noalias() += this->Wst*decState[i]->h; // Set ``s_tilde``
  // sequence and Tree (phrase node)
  contextTree = this->zeros;
  this->calculateAlpha(state, decState[i], alphaTree);
  for (int j = 1; j < (int)state->encState.size(); ++j) {
    contextTree += alphaTree.coeff(j-1, 0)*state->encState[j]->h;
  }
  for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) { 
    contextTree += alphaTree.coeff(j+(int)state->encState.size()-1, 0)*state->encTreeNodeVec[j]->state->h;
  }
  s_tilde += this->Wct*contextTree;
  ActFunc::tanh(s_tilde); // s~tanh(W_c[ht])
}

void AttentionTreeEncDec::candidateDecoder(AttentionTreeEncDec::State* state, std::vector<LSTM::State*>& decState,
					   VecD& s_tilde, const std::vector<int>& tgt,
					   VecD& c0, VecD& s0, const int i) { // For translate()

  if (i == 0) { // initialize decoder's initial state
    if (initDec == AttentionTreeEncDec::TREELSTM) { // Left: Sequence, Right: Tree (Phrase node)
      // copy ``c`` and ``h``
      this->initDecTree.forward((TreeLSTM::State*)decState[i], 
				state->encState.back(), state->encTreeState->state);
    } else if (initDec == AttentionTreeEncDec::CONCAT) { // inspired by (Bahdanau et al., 2015)
      c0 = this->WcellSeq*state->encState.back()->c
	+ this->WcellTree*state->encTreeState->state->c;
      s0 = this->WhSeq*state->encState.back()->h
	+ this->WhTree*state->encTreeState->state->h;
      ActFunc::tanh(c0);
      ActFunc::tanh(s0);
      decState[i]->c = c0; // tanh(W[cell_Seq; cell_Tree])
      decState[i]->h = s0; // tanh(W[h_Seq; h_Tree])
    } else {}
  }
  else { // i >= 1
    if (this->inputFeeding) {
      // input-feeding approach [Luong et al., EMNLP2015]
      this->dec.forward(this->targetEmbed.col(tgt[i-1]), s_tilde, 
			decState[i-1], decState[i]); // (xt, at (use previous ``s_tilde``, prev, cur)
    } else {
      this->dec.forward(this->targetEmbed.col(tgt[i-1]), 
			decState[i-1], decState[i]); // (xt, prev, cur)
    }
  }
}

void AttentionTreeEncDec::readStat(std::unordered_map<int, std::unordered_map<int, Real> >& stat) {
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
}

void AttentionTreeEncDec::translate(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
				    AttentionTreeEncDec::State* state,
				    const int beamSize, const int maxLength, const int showNum) {
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist; 
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionTreeEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // Vector for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)

  this->encoder(src, state);

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      }
      if (i == 0) {
	candidate[j].decState.push_back(new TreeLSTM::State);
      } else {
	candidate[j].decState.push_back(new LSTM::State);
      }
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      this->candidateDecoder(state, candidate[j].decState, candidate[j].s_tilde, candidate[j].tgt, c0, s0, i);
      this->decoderAttention(state, candidate[j].decState, contextTree, alphaTree, candidate[j].s_tilde, i);
      // candidate[j].showAlphaTree.row(i) = alphaTree.transpose();

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
  std::cout << std::endl;

  this->deleteStateList(stateList);

  this->clearState(state, srcParsedFlag);
}

void AttentionTreeEncDec::translate(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
				    std::vector<int>& trans, AttentionTreeEncDec::State* state,
				    const int beamSize, const int maxLength) {
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist;
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionTreeEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // x for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)

  this->encoder(src, state);

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      }
      if (i == 0) { 
	candidate[j].decState.push_back(new TreeLSTM::State);
      } else {
	candidate[j].decState.push_back(new LSTM::State);
      }
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      this->candidateDecoder(state, candidate[j].decState, candidate[j].s_tilde, candidate[j].tgt, c0, s0, i);
      this->decoderAttention(state, candidate[j].decState, contextTree, alphaTree, candidate[j].s_tilde, i);
      // candidate[j].showAlphaTree.row(i) = alphaTree.transpose();

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

  this->makeTrans(candidate[0].tgt, trans);

  this->deleteStateList(stateList);

  this->clearState(state, srcParsedFlag);
}



void AttentionTreeEncDec::translate2(const std::vector<int>& src, const AttentionTreeEncDec::IndexNode* srcParsed, const bool srcParsedFlag,
				     std::vector<int>& trans, AttentionTreeEncDec::State* state,
				     const std::unordered_map<int, std::unordered_map<int, Real> >& stat,
				     const int beamSize, const int maxLength) {
  const Real minScore = -1.0e+05;
  const int srcLen = src.size()-1;
  MatD score(this->targetEmbed.cols(), beamSize);
  VecD targetDist;
  std::vector<int> tgt;
  std::vector<LSTM::State*> stateList;
  std::vector<AttentionTreeEncDec::DecCandidate> candidate(beamSize), candidateTmp(beamSize);
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // x for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)

  for (int j=0; j < beamSize; ++j) {
    candidate[j].showAlphaTree = MatD::Zero(maxLength, alphaSize);
  }

  /* Encoder starts */
  this->encoder(src, state);

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      if (candidate[j].stop) {
	score.col(j).fill(candidate[j].score);
	continue;
      } else{}
      if (i == 0) { 
	candidate[j].decState.push_back(new TreeLSTM::State);
      } else {
	candidate[j].decState.push_back(new LSTM::State);
      }
      stateList.push_back(candidate[j].decState.back()); // ``stateList`` holds a list of the added LSTM units

      this->candidateDecoder(state, candidate[j].decState, candidate[j].s_tilde, candidate[j].tgt, c0, s0, i);
      this->decoderAttention(state, candidate[j].decState, contextTree, alphaTree, candidate[j].s_tilde, i);
      // candidate[j].showAlphaTree.row(i) = alphaTree.transpose();

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
	if (stat.count(srcLen)){
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

  this->makeTrans(candidate[0].tgt, trans);

  this->deleteStateList(stateList);

  this->clearState(state, srcParsedFlag);
}

void AttentionTreeEncDec::showTopAlphaTree(MatD& showAlphaTree, 
					   AttentionTreeEncDec::State* state,
					   const std::vector<int>& src, const std::vector<int>& tgt){
  std::string fileName = "pathToSaveOfAlphaTree"; // TODO: Modify the path
  std::ofstream outputFile;
  outputFile.open(fileName, std::ios::app);

  outputFile << std::endl;
  outputFile << "Src: ";
  for (int i = 0; i < src.size(); ++i){
    outputFile << this->sourceVoc.tokenList[src[i]]->str << " ";
  }
  outputFile << std::endl;
  outputFile << "Tgt: ";
  for (int i = 0; i < tgt.size(); ++i){
    outputFile << this->targetVoc.tokenList[tgt[i]]->str << " ";
  }
  outputFile << std::endl;

  for (int i = 0; i < (int)tgt.size(); ++i) {
    outputFile << "***" << this->targetVoc.tokenList[tgt[i]]->str << ": " << std::endl;
    print(this->targetVoc.tokenList[tgt[i]]->str);
    for (int j = 0, row, col; j < 5; ++j) { // Show Top5
      showAlphaTree.row(i).maxCoeff(&row, &col);
      outputFile << " (" << showAlphaTree.coeff(i, col) << ") ";
      print(showAlphaTree.coeff(i, col));
      if (col < (int)state->encState.size()-1) {
	outputFile << this->sourceVoc.tokenList[src[col]]->str;
	print(this->sourceVoc.tokenList[src[col]]->str);
      }
      else{
	showTreeNode(state->encTreeNodeVec[col+1-(int)state->encState.size()], outputFile);
      }
      showAlphaTree.coeffRef(i, col) = -1.0;
      outputFile << std::endl;
    }
    outputFile << std::endl << std::flush;
    if (this->targetVoc.tokenList[tgt[i]]->str == "*EOS*") break;
  }
  outputFile << "##################################" <<std::endl;
  outputFile.close();
}
  
void AttentionTreeEncDec::train(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, Real& loss, 
				AttentionTreeEncDec::State* state, 
				std::vector<BlackOut::State*>& blackOutState) {
  VecD targetDist;
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // Vector for attentional weight
  // MatD showAlphaTree = MatD::Zero(data->tgt.size(), state->encState.size()-1+state->encTreeNodeVec.size());

  VecD s_tilde; // Decoder; s~
  VecD del_stilde;
  VecD del_contextTree;
  VecD del_alphaTree = VecD::Zero(alphaSize);
  VecD del_alignScore; // delta for alignment score
  VecD del_c0, del_s0;
  loss = 0.0;

  this->encoder(data->src, state);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    this->decoder(data, state, s_tilde, c0, s0, i);
    this->decoderAttention(state, state->decState, contextTree, alphaTree, s_tilde, i);
    //showAlphaTree.push_back(MatD::Zero(data->tgt.size(), 
    //       state->encState.size()-1+state->encTreeNodeVec.size()));
    //showAlphaTree.back().row(i) = alphaTree[i].transpose();

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
    grad.Wct += del_stilde*contextTree.transpose();
    del_contextTree = this->Wct.transpose()*del_stilde;

    // del_contextTree
    for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
      state->encState[j+1]->delh += alphaTree.coeff(j, 0) * del_contextTree;
      del_alphaTree.coeffRef(j, 0) = del_contextTree.dot(state->encState[j+1]->h);
    }
    for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) { // Tree (Phrase node)
      state->encTreeNodeVec[j]->state->delh += alphaTree.coeff(j+(int)data->src.size(), 0) * del_contextTree;
      del_alphaTree.coeffRef(j+(int)data->src.size(), 0) = del_contextTree.dot(state->encTreeNodeVec[j]->state->h);
    }
    del_alignScore = alphaTree.array()*(del_alphaTree.array()-alphaTree.dot(del_alphaTree)); // X.array() - scalar; np.array() -= 1
    if (this->attenType == AttentionTreeEncDec::DOT) { // h^T*s
      for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
	state->encState[j+1]->delh += del_alignScore.coeff(j, 0)*state->decState[i]->h;
	state->decState[i]->delh += del_alignScore.coeff(j, 0)*state->encState[j+1]->h;
      }
      for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) { // Tree (phrase node)
	state->encTreeNodeVec[j]->state->delh += 
	  del_alignScore.coeff(j+(int)data->src.size(), 0)*state->decState[i]->h;
	state->decState[i]->delh += del_alignScore.coeff(j+(int)data->src.size(), 0)*state->encTreeNodeVec[j]->state->h;
      }
    } else if (this->attenType == AttentionTreeEncDec::GENERAL) { // s^T*W*h
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh += (this->WgeneralTree.transpose()*state->decState[i]->h)*del_alignScore.coeff(j, 0);
	state->decState[i]->delh += (this->WgeneralTree*state->encState[j+1]->h)*del_alignScore.coeff(j, 0);
	grad.WgeneralTree += del_alignScore.coeff(j, 0)*state->decState[i]->h*state->encState[j+1]->h.transpose();
      }
      for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) {
	state->encTreeNodeVec[j]->state->delh += 
	  (this->WgeneralTree.transpose()*state->decState[i]->h)*del_alignScore.coeff(j+(int)data->src.size(), 0);
	state->decState[i]->delh += 
	  (this->WgeneralTree*state->encTreeNodeVec[j]->state->h)*del_alignScore.coeff(j+(int)data->src.size(), 0);
	grad.WgeneralTree += 
	  del_alignScore.coeff(j+(int)data->src.size(), 0)*state->decState[i]->h*state->encTreeNodeVec[j]->state->h.transpose();
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
  if (initDec == AttentionTreeEncDec::TREELSTM) {
    this->initDecTree.backward((TreeLSTM::State*)state->decState[0], state->encState.back(), state->encTreeState->state, 
			       grad.treeLstmInitDecGrad);
  } else if (initDec == AttentionTreeEncDec::CONCAT) {
    del_c0 = state->decState[0]->delc;
    del_s0 = state->decState[0]->delh;
    del_c0.array() *= ActFunc::tanhPrime(c0).array();
    del_s0.array() *= ActFunc::tanhPrime(s0).array();
    state->encState.back()->delc += this->WcellSeq.transpose()*del_c0;
    grad.WcellSeq += del_c0*state->encState.back()->c.transpose();
    state->encTreeState->state->delc += this->WcellTree.transpose()*del_c0;
    grad.WcellTree += del_c0*state->encTreeState->state->c.transpose();
    state->encState.back()->delh += this->WhSeq.transpose()*del_s0;
    grad.WhSeq += del_s0*state->encState.back()->h.transpose();
    state->encTreeState->state->delh += this->WhTree.transpose()*del_s0;
    grad.WhTree += del_s0*state->encTreeState->state->h.transpose();
  } else {}

  // backprop through structure
  if (data->srcParsedFlag) {
    state->leafCounter = 0;
    this->backpropThroughStructure(state->encTreeState, 
				   state->encTreeState->left, state->encTreeState->right, 
				   grad, state);
  } else {}
  // Encoder Backpropagation after adding leaves' gradients (this is already done in ``backpropThroughStructure``)
  for (int i = data->src.size(); i >= 1; --i) {
    if (i == 1) {
      this->enc.backward(state->encState[i], grad.lstmSrcGrad, 
			 this->sourceEmbed.col(data->src[i-1]));
    } else {
      this->enc.backward(state->encState[i-1], state->encState[i], grad.lstmSrcGrad, 
			 this->sourceEmbed.col(data->src[i-1]));
    }
    if (grad.sourceEmbed.count(data->src[i-1])) {
      grad.sourceEmbed.at(data->src[i-1]) += state->encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = state->encState[i]->delx;
    } 
  }
}

void AttentionTreeEncDec::train(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, Real& loss, 
				AttentionTreeEncDec::State* state, 
				std::vector<BlackOut::State*>& blackOutState,
				std::vector<VecD>& s_tilde, std::vector<VecD>& del_stilde) {
  VecD targetDist;
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  MatD alphaTree = MatD(alphaSize, data->tgt.size());
  // std::vector<VecD> showAlphaTree;

  VecD del_contextTree;
  VecD del_alphaTree = VecD::Zero(alphaSize);
  VecD del_alignScore; // delta for alignment score
  VecD del_c0, del_s0;
  loss = 0.0;

  this->encoder(data->src, state);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    if (i == 0) {
      if (initDec == AttentionTreeEncDec::TREELSTM) { // Left: Sequence, Right: Tree (Phrase node)
	// copy ``c`` and ``h`` and set them into ``initialState`` 
	this->initDecTree.forward((TreeLSTM::State*)state->decState[i], 
				  state->encState.back(), state->encTreeState->state);
      } else if (initDec == AttentionTreeEncDec::CONCAT) { //inspired 
	c0 = this->WcellSeq*state->encState.back()->c
	  + this->WcellTree*state->encTreeState->state->c;
	s0 = this->WhSeq*state->encState.back()->h
	  + this->WhTree*state->encTreeState->state->h;
	ActFunc::tanh(c0);
	ActFunc::tanh(s0);
	state->decState[i]->c = c0; // tanh(W[cell_Seq; cell_Tree])
	state->decState[i]->h = s0; // tanh(W[h_Seq; h_Tree])
      } else {}
    } else { // i >= 1
      // input-feeding approach [Luong et al., EMNLP2015]
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), s_tilde[i-1], 
			state->decState[i-1], state->decState[i]); // (xt, at, prev, cur)
    }

    /* Attention */
    s_tilde[i] = this->bs;
    s_tilde[i].noalias() += this->Wst*state->decState[i]->h;
    // sequence and Tree (phrase node)
    contextTree = this->zeros; // initialize
    // Matrix for attentional weight
    //showAlphaTree.push_back(MatD::Zero(data->tgt.size(), 
    //       state->encState.size()-1+state->encTreeNodeVec.size()));
    this->calculateAlpha(state, state->decState[i], alphaTree, i); // (!) Insert the i-th alphaTree
    //showAlphaTree.back().row(i) = alphaTree[i].transpose();
    for (int j = 1; j < (int)state->encState.size(); ++j) {
      contextTree.noalias() += alphaTree.coeff(j-1, i)*state->encState[j]->h;
    }
    for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) {
      contextTree.noalias() += alphaTree.coeff(j+(int)state->encState.size()-1, i)*state->encTreeNodeVec[j]->state->h;
    }
    s_tilde[i].noalias() += this->Wct*contextTree;
    ActFunc::tanh(s_tilde[i]); // s~tanh(W_c[ht; c_Tree])

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
  for (int i = (int)data->tgt.size()-1; i >= 0; --i) {
    /* Attention's Backpropagation */
    // del_stilde 
    if (i < (int)data->tgt.size()-1) {
      del_stilde[i] += state->decState[i+1]->dela; // add gradients to the previous del_stilde 
                                                   // by input-feeding [Luong et al., EMNLP2015]
    } else {}
    del_stilde[i].array() *= ActFunc::tanhPrime(s_tilde[i]).array();
    state->decState[i]->delh.noalias() += this->Wst.transpose()*del_stilde[i];
    grad.Wst.noalias() += del_stilde[i]*state->decState[i]->h.transpose();
    grad.bs += del_stilde[i];
    grad.Wct.noalias() += del_stilde[i]*contextTree.transpose();
    del_contextTree.noalias() = this->Wct.transpose()*del_stilde[i];
    // del_contextTree
    for (int j = 0; j < (int)data->src.size(); ++j) { // Seq
      state->encState[j+1]->delh.noalias() += alphaTree.coeff(j, i) * del_contextTree;
      del_alphaTree.coeffRef(j, 0) = del_contextTree.dot(state->encState[j+1]->h);
    }
    for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) { // Tree (Phrase node)
      state->encTreeNodeVec[j]->state->delh.noalias() += alphaTree.coeff(j+(int)data->src.size(), i) * del_contextTree;
      del_alphaTree.coeffRef(j+(int)data->src.size(), 0) = del_contextTree.dot(state->encTreeNodeVec[j]->state->h);
    }
    del_alignScore = alphaTree.col(i).array()*(del_alphaTree.array()-alphaTree.col(i).dot(del_alphaTree)); // X.array() - scalar; np.array() -= 1
    if (this->attenType == AttentionTreeEncDec::DOT) { // h^T*s
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh.noalias() += del_alignScore.coeff(j, 0)*state->decState[i]->h;
	state->decState[i]->delh.noalias() += del_alignScore.coeff(j, 0)*state->encState[j+1]->h;
      }
      for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) {
	state->encTreeNodeVec[j]->state->delh.noalias() += 
	  del_alignScore.coeff(j+(int)data->src.size(), 0)*state->decState[i]->h;
	state->decState[i]->delh.noalias() += del_alignScore.coeff(j+(int)data->src.size(), 0)*state->encTreeNodeVec[j]->state->h;
      }
    } else if (this->attenType == AttentionTreeEncDec::GENERAL) { // s^T*W*h
      for (int j = 0; j < (int)data->src.size(); ++j) {
	state->encState[j+1]->delh += (this->WgeneralTree.transpose()*state->decState[i]->h)*del_alignScore.coeff(j, 0);
	state->decState[i]->delh += (this->WgeneralTree*state->encState[j+1]->h)*del_alignScore.coeff(j, 0);
	grad.WgeneralTree += del_alignScore.coeff(j, 0)*state->decState[i]->h*state->encState[j+1]->h.transpose();
      }
      for (int j = 0; j < (int)state->encTreeNodeVec.size(); ++j) {
	state->encTreeNodeVec[j]->state->delh += 
	  (this->WgeneralTree.transpose()*state->decState[i]->h)*del_alignScore.coeff(j+(int)data->src.size(), 0);
	state->decState[i]->delh += 
	  (this->WgeneralTree*state->encTreeNodeVec[j]->state->h)*del_alignScore.coeff(j+(int)data->src.size(), 0);
	grad.WgeneralTree += 
	  del_alignScore.coeff(j+(int)data->src.size(), 0)*state->decState[i]->h*state->encTreeNodeVec[j]->state->h.transpose();
      }
    } else {}
    if (i > 0) {
      state->decState[i-1]->delc = this->zeros;
      state->decState[i-1]->delh = this->zeros;
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
  if (initDec == AttentionTreeEncDec::TREELSTM) {
    this->initDecTree.backward((TreeLSTM::State*)state->decState[0], state->encState.back(), state->encTreeState->state, 
			       grad.treeLstmInitDecGrad);
  } else if (initDec == AttentionTreeEncDec::CONCAT) {
    del_c0 = state->decState[0]->delc;
    del_s0 = state->decState[0]->delh;
    del_c0.array() *= ActFunc::tanhPrime(c0).array();
    del_s0.array() *= ActFunc::tanhPrime(s0).array();
    state->encState.back()->delc += this->WcellSeq.transpose()*del_c0;
    grad.WcellSeq += del_c0*state->encState.back()->c.transpose();
    state->encTreeState->state->delc += this->WcellTree.transpose()*del_c0;
    grad.WcellTree += del_c0*state->encTreeState->state->c.transpose();
    state->encState.back()->delh += this->WhSeq.transpose()*del_s0;
    grad.WhSeq += del_s0*state->encState.back()->h.transpose();
    state->encTreeState->state->delh += this->WhTree.transpose()*del_s0;
    grad.WhTree += del_s0*state->encTreeState->state->h.transpose();
  } else {}

  //backprop through structure
  if (data->srcParsedFlag) {
    state->leafCounter = 0;
    this->backpropThroughStructure(state->encTreeState, 
				   state->encTreeState->left, state->encTreeState->right, 
				   grad, state);
  } else {}
  // Encoder Backpropagation while adding leaves' gradients (this is already done in ``backpropThroughStructure``)
  for (int i = data->src.size(); i >= 1; --i) {
    if (i == 1) {
      this->enc.backward(state->encState[i], grad.lstmSrcGrad, 
			 this->sourceEmbed.col(data->src[i-1]));
    } else {
      this->enc.backward(state->encState[i-1], state->encState[i], grad.lstmSrcGrad,
			 this->sourceEmbed.col(data->src[i-1]));
    }
    if (grad.sourceEmbed.count(data->src[i-1])) {
      grad.sourceEmbed.at(data->src[i-1]) += state->encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = state->encState[i]->delx;
    }
  }
}

void AttentionTreeEncDec::calculateAlpha(const AttentionTreeEncDec::State* state,
					 const LSTM::State* decState, VecD& alphaTree) { // calculate attentional weight;
  /* ``decState`` is a encNodeVec itsself and ``encTreeNodeVec`` for TreeLSTM units */
  const int encStateSize = state->encState.size();
  const int encTreeSize = state->encTreeNodeVec.size();
  if (this->attenType  == AttentionTreeEncDec::DOT) { //inner product; h^T*s
    // Sequnce (leaf)
    for (int i = 1; i < encStateSize; ++i) { // encState is 1-origin (encState[0] == h0)
      alphaTree.coeffRef(i-1, 0) = state->encState[i]->h.dot(decState->h); // coeffRef: insertion
    }
    // Tree (phrase node)
    for (int i = 0; i < encTreeSize; ++i) { // encState[0]=h0
      alphaTree.coeffRef(i+encStateSize-1, 0) = 
	state->encTreeNodeVec[i]->state->h.dot(decState->h); // coeffRef: insertion
    }
  }
  else if (this->attenType == AttentionTreeEncDec::GENERAL) { // s^T*W*h
    for (int i = 1; i < (int)state->encState.size(); ++i) {
      alphaTree.coeffRef(i-1, 0) = decState->h.dot(this->WgeneralTree * state->encState[i]->h);
    }
    for (int i = 0; i < (int)state->encTreeNodeVec.size(); ++i) {
      alphaTree.coeffRef(i+(int)state->encState.size()-1, 0) = 
	decState->h.dot(this->WgeneralTree * state->encTreeNodeVec[i]->state->h);
    }
  } else {}
  // softmax of ``alphaTree``
  alphaTree.array() -= alphaTree.maxCoeff(); // stable softmax
  alphaTree = alphaTree.array().exp(); // exp() operation for all elements; np.exp(alphaTree) 
  alphaTree /= alphaTree.array().sum(); // alphaTree.sum()
}

void AttentionTreeEncDec::calculateAlpha(const AttentionTreeEncDec::State* state,
					 const LSTM::State* decState, MatD& alphaTree, const int colNum) { // calculate attentional weight;
  /* ``decState`` is a encNodeVec itsself and ``encTreeNodeVec`` for TreeLSTM units */
  const int encStateSize = state->encState.size();
  const int encTreeSize = state->encTreeNodeVec.size();
  if (this->attenType  == AttentionTreeEncDec::DOT) { // inner product: h^T*s
    // Sequnce (leaf)
    for (int i = 1; i < encStateSize; ++i) { // encState is 1-origin (encState[0] == h0)
      alphaTree.coeffRef(i-1, colNum) = state->encState[i]->h.dot(decState->h); // coeffRef: insertion
    }
    // Tree (phrase node)
    for (int i = 0; i < encTreeSize; ++i) { // encState[0]=h0
      alphaTree.coeffRef(i+encStateSize-1, colNum) = 
	state->encTreeNodeVec[i]->state->h.dot(decState->h); // coeffRef: insertion
    }
  }
  else if (this->attenType == AttentionTreeEncDec::GENERAL) { // s^T*W*h
    for (int i = 1; i < (int)state->encState.size(); ++i) {
      alphaTree.coeffRef(i-1, colNum) = decState->h.dot(this->WgeneralTree * state->encState[i]->h);
    }
    for (int i = 0; i < (int)state->encTreeNodeVec.size(); ++i) {
      alphaTree.coeffRef(i+(int)state->encState.size()-1, colNum) = 
	decState->h.dot(this->WgeneralTree * state->encTreeNodeVec[i]->state->h);
    }
  } else {}
  // softmax of ``alphaTree``
  alphaTree.col(colNum).array() -= alphaTree.col(colNum).maxCoeff(); // stable softmax
  alphaTree.col(colNum) = alphaTree.col(colNum).array().exp(); // exp() operation for all elements; np.exp(alphaTree) 
  alphaTree.col(colNum) /= alphaTree.col(colNum).array().sum(); // alphaTree.sum()
}

void AttentionTreeEncDec::sgd(const AttentionTreeEncDec::Grad& grad, const Real learningRate) {
  this->Wst -= learningRate * grad.Wst;
  this->bs -= learningRate * grad.bs;
  this->Wct -= learningRate * grad.Wct;

  this->WgeneralTree -= learningRate * grad.WgeneralTree;
  this->WcellSeq -= learningRate * grad.WcellSeq;
  this->WcellTree -= learningRate * grad.WcellTree;
  this->WhSeq -= learningRate * grad.WhSeq;
  this->WhTree -= learningRate * grad.WcellTree;

  ///* TODO: Freezing embed
  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    this->sourceEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it) {
    this->targetEmbed.col(it->first) -= learningRate * it->second;
  }
}

void AttentionTreeEncDec::backpropThroughStructure(AttentionTreeEncDec::StateNode* parent, 
						   AttentionTreeEncDec::StateNode* left, 
						   AttentionTreeEncDec::StateNode* right, 
						   AttentionTreeEncDec::Grad& grad, 
						   AttentionTreeEncDec::State* state) {
  //leaf
  if (parent->tokenIndex >= 0) { // Add leaf's gradients to the corresponding Sequential state's gradients
    ++(state->leafCounter);
    state->encState[state->leafCounter]->delc += parent->state->delc;
    state->encState[state->leafCounter]->delh += parent->state->delh; 
    return;
  }
  //non-leaf
  else {
    this->encTree.backward((TreeLSTM::State*)parent->state, left->state, right->state, grad.treeLstmEncGrad);
    this->backpropThroughStructure(left, left->left, left->right, grad, state);
    this->backpropThroughStructure(right, right->left, right->right, grad, state);
  }
}

void AttentionTreeEncDec::trainOpenMP() { // OpenMP Multi-threading
  static std::vector<AttentionTreeEncDec::ThreadArg*> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static AttentionTreeEncDec::Grad grad;
  Real lossTrain = 0.0, lossDev = 0.0, tgtNum = 0.0;
  Real gradNorm, lr = this->learningRate;
  struct timeval start, end;
  static float countModel = -0.5;

  if (args.empty()) {
    for (int i = 0; i < this->threadNum; ++i) {
      args.push_back(new AttentionTreeEncDec::ThreadArg(*this));
    }
    for (int i = 0, step = this->trainData.size()/this->miniBatchSize; i< step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->miniBatchSize-1)));
      // Create pairs of MiniBatch, e.g. [(0,3), (4, 7), ...]
    }
    // The whole gradients
    grad.lstmSrcGrad = LSTM::Grad(this->enc);
    grad.lstmTgtGrad = LSTM::Grad(this->dec);
    grad.treeLstmEncGrad = TreeLSTM::Grad(this->encTree);
    grad.treeLstmInitDecGrad = TreeLSTM::Grad(this->initDecTree);
    grad.softmaxGrad = SoftMax::Grad(this->softmax);

    grad.Wst = MatD::Zero(this->Wst.rows(), this->Wst.cols());
    grad.bs = VecD::Zero(this->bs.size());
    grad.Wct = MatD::Zero(this->Wct.rows(), this->Wct.cols());
    grad.WgeneralTree= MatD::Zero(this->WgeneralTree.rows(), this->WgeneralTree.cols());
    grad.WcellSeq= MatD::Zero(this->WcellSeq.rows(), this->WcellSeq.cols());
    grad.WcellTree= MatD::Zero(this->WcellTree.rows(), this->WcellTree.cols());
    grad.WhSeq= MatD::Zero(this->WhSeq.rows(), this->WhSeq.cols());
    grad.WhTree= MatD::Zero(this->WhTree.rows(), this->WhTree.cols());
  }

  gettimeofday(&start, 0);
  this->rnd.shuffle(this->trainData);

  int count = 0;

  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it) {
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

    std::unordered_map<int, AttentionTreeEncDec::State*> stateList;
    for (int j = it->first; j <= it->second; ++j) {
      stateList[j] = new AttentionTreeEncDec::State;
      this->makeState(this->trainData[j], stateList.at(j));
    } // prepare states before omp parallel

#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic) shared(args, stateList)
    for (int i = it->first; i <= it->second; ++i) {
      int id = omp_get_thread_num();
      Real loss;
      if(this->inputFeeding) {
	this->train(this->trainData[i], args[id]->grad, loss, stateList[i], args[id]->blackOutState, args[id]->s_tilde, args[id]->del_stilde);
      } else {
	this->train(this->trainData[i], args[id]->grad, loss, stateList[i], args[id]->blackOutState);
      }

      /* ..Gradient Checking.. :) */
      //this->gradChecker(this->trainData[i], this->dec.Whi, args[id]->grad.lstmTgtGrad.Whi, stateList[i], args[id]->blackOutState);
      //this->gradChecker(this->trainData[i], args[id]->grad, stateList[i], args[id]->blackOutState);

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
    this->encTree.sgd(grad.treeLstmEncGrad, lr);
    this->initDecTree.sgd(grad.treeLstmInitDecGrad, lr);
    if (!this->useBlackOut) {
      this->softmax.sgd(grad.softmaxGrad, lr);
    }
    else {
      this->blackOut.sgd(grad.blackOutGrad, lr);
    }
    this->sgd(grad, lr);

    grad.init();

    if (count == (int)(miniBatch.size()/2)) { // saveModel after halving epoch
      this->saveModel(countModel);
      countModel += 1.;
    }
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

void AttentionTreeEncDec::showTreeNode(const AttentionTreeEncDec::StateNode* node) {
  if (node->tokenIndex >= 0) {
    std::cout << this->sourceVoc.tokenList[node->tokenIndex]->str << " ";
  }
  else {
    this->showTreeNode(node->left);
    this->showTreeNode(node->right);
  }
}


void AttentionTreeEncDec::showTreeNode(const AttentionTreeEncDec::StateNode* node, std::ofstream& outputFile) {
  if (node->tokenIndex >= 0) {
    outputFile << this->sourceVoc.tokenList[node->tokenIndex]->str << " ";
  }
  else {
    this->showTreeNode(node->left, outputFile);
    this->showTreeNode(node->right, outputFile);
  }
}

void AttentionTreeEncDec::makeTreeState(const AttentionTreeEncDec::IndexNode* node, 
					AttentionTreeEncDec::StateNode* stateNode, 
					AttentionTreeEncDec::State* state) {
  // leaf
  if (node->tokenIndex >= 0) { // set a Sequential state
    stateNode->tokenIndex = node->tokenIndex;
    state->encTreeLeafVec.push_back(stateNode); // Vecotr for Tree (Sequence; leaf nodes)
    return;
  }
  // non-leaf
  else {
    state->encTreeNodeVec.push_back(stateNode); // for paying attention to Tree (phrase node)
    stateNode->left = new AttentionTreeEncDec::StateNode(-1, new TreeLSTM::State, 0, 0);
    this->makeTreeState(node->left, stateNode->left, state);
    stateNode->right = new AttentionTreeEncDec::StateNode(-1, new TreeLSTM::State, 0, 0);
    this->makeTreeState(node->right, stateNode->right, state);
  }
}

void AttentionTreeEncDec::deleteTreeState(AttentionTreeEncDec::StateNode* stateNode) {
  // leaf
  if (stateNode->tokenIndex >= 0) {
    stateNode->state->clear();
    if (stateNode->state != NULL) {
      delete stateNode->state;
      stateNode->state = NULL;
    }
    return;
  }
  // non-leaf
  else {
    stateNode->state->clear();
    if (stateNode->state != NULL) {
      delete stateNode->state;
      stateNode->state = NULL;
    }
    this->deleteTreeState(stateNode->left);
    this->deleteTreeState(stateNode->right);
    if (stateNode->left != NULL) {
      delete stateNode->left;
      stateNode->left = NULL;
    }
    if (stateNode->right != NULL) {
      delete stateNode->right;
      stateNode->right = NULL;
    }
  }
}

void AttentionTreeEncDec::clearTreeState(AttentionTreeEncDec::StateNode* stateNode) {
  // leaf
  if (stateNode->tokenIndex >= 0) {
    stateNode->state->clear();
    return;
  }
  // non-leaf
  else {
    stateNode->state->clear();
    this->clearTreeState(stateNode->left);
    this->clearTreeState(stateNode->right);
  }
}

void AttentionTreeEncDec::makeSubTree(const boost::property_tree::ptree& pt, 
				      AttentionTreeEncDec::IndexNode* node, 
				      std::vector<int>& src) {
  // Enju is a Binary Tree; We found the words of a sentence (Enju's xml data) by a depth-first seach algorithm!
  /* Cons->Cons(R) ->Cons(L); Cons->Cons, Cons->Tok(=Tok) */
  int numNode = 0;
  numNode = pt.size() - pt.count("<xmlattr>");
  if (numNode == 2) { // two childlen nodes; (cons, cons)
    int right = 0;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &child_, pt.get_child("")) {
      if (boost::lexical_cast<std::string>(child_.first.data()) == "cons") {
	if (right < 1) { // left node
	  node->left = new AttentionTreeEncDec::IndexNode(-1, 0, 0);
	  this->makeSubTree(child_.second, node->left, src);
	}
	else { // right node
	  node->right = new AttentionTreeEncDec::IndexNode(-1, 0, 0);
	  this->makeSubTree(child_.second, node->right, src);
	}
	++right;
      }
      else { //xmlattr
      }
    }
  }
  else if (numNode == 1) { // one child node
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &child_, pt.get_child("")) {
      if (boost::lexical_cast<std::string>(child_.first.data()) == "tok") { // Convert (cons -> tok) => (node->tok)
	boost::optional<std::string> tok_ = boost::lexical_cast<std::string>(child_.second.data());
	std::string str = tok_.get();
	std::transform(str.begin(), str.end(), str.begin(), tolower);
	int tokenIndex = this->sourceVoc.tokenIndex.count(str) ? this->sourceVoc.tokenIndex.at(str) : this->sourceVoc.unkIndex;
	node->tokenIndex = tokenIndex;
	if (src[this->leafCounter] == tokenIndex) { // .. Check the correspondence .. ;)
	  ++(this->leafCounter);
	} else {
	  print("Error: the index of ``src`` doens't match the one of ``encTreeLeafVec``. You should make a mapping table for them.");
	  exit(1);
	}
	return;
      }
      else if (boost::lexical_cast<std::string>(child_.first.data()) == "cons") {
	this->makeSubTree(child_.second, node, src);
      } else { }
    }
  } else {}
}

void AttentionTreeEncDec::makeTree(const std::string& srcParsed, AttentionTreeEncDec::Data* data) {
  AttentionTreeEncDec::IndexNode* node;
  boost::property_tree::ptree pt;
  std::istringstream input(srcParsed);
  this->leafCounter = 0; // Check the correspondence between the leaf (encTreeLeafVec; 0-origin)
                         // ............................ and the index (src; 0-origin)

  boost::property_tree::xml_parser::read_xml(input, pt);

  boost::optional<std::string> parseStatus = pt.get_optional<std::string>("sentence.<xmlattr>.parse_status");
  if (parseStatus.get() == "success") {
    data->srcParsedFlag = true;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &root_, pt.get_child("sentence")) {
      if (boost::lexical_cast<std::string>(root_.first.data()) == "cons") {
	node = new AttentionTreeEncDec::IndexNode(-1, 0, 0);
	data->srcParsed = node; //root
	this->makeSubTree(root_.second, node, data->src);
      }
    }
    boost::optional<std::string> pp = pt.get_optional<std::string>("sentence");
  }
  else if (parseStatus.get() == "fragmental parse") {
    data->srcParsedFlag = false;
    node = new AttentionTreeEncDec::IndexNode(-1, 0, 0);
    data->srcParsed = node; //root
  } 
  else {
    data->srcParsedFlag = false;
    node = new AttentionTreeEncDec::IndexNode(-1, 0, 0);
    data->srcParsed = node; //root
  }
}

Real AttentionTreeEncDec::calcLoss(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state) {
  VecD targetDist;
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  MatD showAlphaTree = MatD::Zero(data->tgt.size(), alphaSize);
  VecD alphaTree = VecD(alphaSize); // Vector for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real loss = 0.;

  this->encoder(data->src, state);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    this->decoder(data, state, s_tilde, c0, s0, i);
    this->decoderAttention(state, state->decState, contextTree, alphaTree, s_tilde, i);
    //showAlphaTree.push_back(MatD::Zero(data->tgt.size(), 
    //       state->encState.size()-1+state->encTreeNodeVec.size()));
    //showAlphaTree.back().row(i) = alphaTree[i].transpose();

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    }
    else {
      this->blackOut.calcDist(s_tilde, targetDist); //Softmax
      loss += this->blackOut.calcLoss(targetDist, data->tgt[i]); // Softmax
    }
  }

  this->clearState(state, data->srcParsedFlag);
  return loss;
}

Real AttentionTreeEncDec::calcPerplexity(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state) {
  // Read only if Parsed status = "success"
  // calculate the perplexity without `*EOS*`
  VecD targetDist;
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // Vector for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real perp = 0.; // Perplexity

  this->encoder(data->src, state);

  for (int i = 0; i < (int)data->tgt.size()-1; ++i) { // omit `*EOS*` in data->tgt
    this->decoder(data, state, s_tilde, c0, s0, i);
    this->decoderAttention(state, state->decState, contextTree, alphaTree, s_tilde, i);

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
    }
    else {
      this->blackOut.calcDist(s_tilde, targetDist); //Softmax
    }
    perp -= log(targetDist.coeff(data->tgt[i], 0)); // Perplexity
  }
  this->clearState(state, data->srcParsedFlag);
  return exp(perp/(data->tgt.size()-1)); // Perplexity without `*EOS*`
}

Real AttentionTreeEncDec::calcLoss(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state,
				   std::vector<BlackOut::State*>& blackOutState) { // for gradient checker
  // Read only if Parsed status = "success"
  VecD targetDist;
  VecD c0, s0; // cell memory and hidden state for initial Decorder (initDec == CONCAT)
  const int alphaSize = state->encState.size()-1+state->encTreeNodeVec.size();
  VecD alphaTree = VecD(alphaSize); // Vector for attentional weight
  VecD contextTree; // C_Tree = Σ (alpha * hidden state)
  VecD s_tilde; // Decoder; s~
  Real loss = 0.0;

  this->encoder(data->src, state);

  for (int i = 0; i < (int)data->tgt.size(); ++i) {
    this->decoder(data, state, s_tilde, c0, s0, i);
    this->decoderAttention(state, state->decState, contextTree, alphaTree, s_tilde, i);

    if (!this->useBlackOut) {
      this->softmax.calcDist(s_tilde, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    }
    else {
      this->blackOut.calcSampledDist(s_tilde, targetDist, *(blackOutState[i]));
      loss += this->blackOut.calcSampledLoss(targetDist);
    }
  }
  this->clearState(state, data->srcParsedFlag);
  return loss;
}
  
void AttentionTreeEncDec::gradChecker(AttentionTreeEncDec::Data* data, MatD& param, const MatD& grad,
				      AttentionTreeEncDec::State* state,
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

void AttentionTreeEncDec::gradChecker(AttentionTreeEncDec::Data* data, AttentionTreeEncDec::Grad& grad, 
				      AttentionTreeEncDec::State* state, 
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

void AttentionTreeEncDec::makeState(const AttentionTreeEncDec::Data* data, AttentionTreeEncDec::State* state) {
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
  TreeLSTM::State *treeLstmState(NULL);
  treeLstmState = new TreeLSTM::State;
  state->decState.push_back(treeLstmState);
  for (auto it = data->tgt.begin()+1; it != data->tgt.end(); ++it) { // includes "End of Sentence" mark
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    state->decState.push_back(lstmState);
  }
  // srcParsed, encTreeNodeVec, encTreeLeafVec

  if (data->srcParsedFlag) {
    state->encTreeState = new AttentionTreeEncDec::StateNode(-1, new TreeLSTM::State, 0, 0);
    this->makeTreeState(data->srcParsed, state->encTreeState, state);
  } else { // zero initialization
    state->encTreeState = initTreeEncState;
  }
}

void AttentionTreeEncDec::deleteState(AttentionTreeEncDec::State* state) {
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
  // srcParsed
  this->deleteTreeState(state->encTreeState);
  if (state->encTreeState != NULL) {
    delete state->encTreeState;
    state->encTreeState = NULL;
  }
  // encTreeNodeVec, encTreeLeafVec
  Utils::swap(state->encTreeNodeVec);
  Utils::swap(state->encTreeLeafVec);
}

void AttentionTreeEncDec::deleteState(std::vector<BlackOut::State*>& blackOutState) {
  // BlackOut
  for (auto it = blackOutState.begin(); it != blackOutState.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(blackOutState);
}

void AttentionTreeEncDec::clearState(AttentionTreeEncDec::State* state, const bool srcParsedFlag) {
  // Src
  for (auto it = state->encState.begin(); it != state->encState.end(); ++it) { // includes "End of Sentence" mark
    (*it)->clear();
  }
  // Tgt
  for (auto it = state->decState.begin(); it != state->decState.end(); ++it) { // includes "End of Sentence" mark
    (*it)->clear();
  }
  // srcParsed
  if (srcParsedFlag) {
    this->clearTreeState(state->encTreeState);
  } else {}
}

void AttentionTreeEncDec::deleteStateList(std::vector<LSTM::State*>& stateList) {
  // Used in translate()
  for (auto it = stateList.begin(); it != stateList.end(); ++it) {
    if (*it != NULL) {
      delete *it;
      *it = NULL;
    }
  }
  Utils::swap(stateList); // Initialize decState vector
}

void AttentionTreeEncDec::makeTrans(std::vector<int>& tgt, std::vector<int>& trans) {
  for (auto it = tgt.begin(); it != tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }
}

void AttentionTreeEncDec::loadCorpus(const std::string& src, const std::string& tgt, const std::string& srcParsed, 
    std::vector<AttentionTreeEncDec::Data*>& data) {
  std::ifstream ifsSrc(src.c_str());
  std::ifstream ifsTgt(tgt.c_str());
  std::ifstream ifsSrcParsed(srcParsed.c_str());

  assert(ifsSrc);
  assert(ifsTgt);
  assert(ifsSrcParsed);

  int numLine = 0;
  // Src
  for (std::string line; std::getline(ifsSrc, line);) {
    std::vector<std::string> tokens;
    data.push_back(new AttentionTreeEncDec::Data);
    Utils::split(line, tokens); // split tokens with the space

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex); // word
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

  // srcParsed
  numLine = 0;
  for (std::string line; std::getline(ifsSrcParsed, line); ) { // "Segmentation Fault" if you read the same sentence?
    this->makeTree(line, data[numLine]); // attentionTreeEncDec
    ++numLine;
  }
}

void AttentionTreeEncDec::saveModel(const float i) {
  std::ostringstream oss;
  std::string parsedMode;

  if (this->parsedTree == AttentionTreeEncDec::PARSED) {
    parsedMode = "PARSED";
  } else { parsedMode = "NONE";}

  oss << this->saveDirName << "Model_AttentionTreeEncDec"
      << ".itr_" << i+1
      << ".ParsedMode_" << parsedMode
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

void AttentionTreeEncDec::saveResult(const Real value, const std::string& name) {
  /* For Model Analysis */
  std::ofstream valueFile;
  std::ostringstream ossValue;
  ossValue << this->saveDirName << "Model_AttentionTreeEncDec" << name;

  valueFile.open(ossValue.str(), std::ios::app); // open a file with 'a' mode

  valueFile << value << std::endl;
}

void AttentionTreeEncDec::demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcParsedTrain, 
			       const std::string& srcDev, const std::string& tgtDev, const std::string& srcParsedDev,
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
  std::vector<AttentionTreeEncDec::Data*> trainData, devData;

  AttentionTreeEncDec attentionTreeEncDec(sourceVoc, targetVoc, trainData, devData, 
					  inputDim, hiddenDim, scale,
					  useBlackOut, blackOutSampleNum, blackOutAlpha,
					  AttentionTreeEncDec::DOT, 
					  AttentionTreeEncDec::PARSED,
					  AttentionTreeEncDec::TREELSTM, 
					  reversed, biasOne, clipThreshold,
					  beamSize, maxGeneNum, 
					  miniBatchSize, threadNum,
					  learningRate, learningRateSchedule, 
					  srcVocaThreshold,
					  inputFeeding,
					  saveDirName);

  attentionTreeEncDec.loadCorpus(srcTrain, tgtTrain, srcParsedTrain, trainData);
  attentionTreeEncDec.loadCorpus(srcDev, tgtDev, srcParsedDev, devData); 
  for (int i = 0; i < (int)devData.size(); ++i) { // make AttentionTreeEncDec::State* for devData
    devData[i]->state = new AttentionTreeEncDec::State;
    attentionTreeEncDec.makeState(devData[i], devData[i]->state);
  }

  auto test = trainData[0];
  test->state = new AttentionTreeEncDec::State;
  test->state->encTreeState = new AttentionTreeEncDec::StateNode(-1, new TreeLSTM::State, 0, 0);
  attentionTreeEncDec.makeState(test, test->state);
  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  for (int i = 0; i < 100; ++i) {
    if (attentionTreeEncDec.learningRateSchedule && i > 4) attentionTreeEncDec.learningRate *= 0.5;
    std::cout << "\nEpoch " << i+1 << " (lr = " << attentionTreeEncDec.learningRate << ")" << std::endl;
    attentionTreeEncDec.trainOpenMP();
    // Save a model
    attentionTreeEncDec.saveModel(i);
    std::cout << "** Greedy Search" << std::endl;
    attentionTreeEncDec.translate(test->src, test->srcParsed, test->srcParsedFlag,
				  test->state, 1, attentionTreeEncDec.maxGeneNum, 1);
    std::cout << "** Beam Search" << std::endl;
    attentionTreeEncDec.translate(test->src, test->srcParsed, test->srcParsedFlag,
				  test->state, attentionTreeEncDec.beamSize, attentionTreeEncDec.maxGeneNum, 5);
  }
}

void AttentionTreeEncDec::evaluate(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcParsedTrain, 
				   const std::string& srcTest, const std::string& tgtTest, const std::string& srcParsedTest, 
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
  static std::vector<AttentionTreeEncDec::Data*> trainData, testData;

  static AttentionTreeEncDec attentionTreeEncDec(sourceVoc, targetVoc, trainData, testData, 
						 inputDim, hiddenDim, scale,
						 useBlackOut, blackOutSampleNum, blackOutAlpha, 
						 AttentionTreeEncDec::DOT, 
						 AttentionTreeEncDec::PARSED,
						 AttentionTreeEncDec::TREELSTM, 
						 reversed, false, 3.0,
						 beamSize, maxGeneNum, 
						 miniBatchSize, threadNum,
						 learningRate, learningRateSchedule,
						 srcVocaThreshold,
						 inputFeeding,
						 saveDirName);

  if (testData.empty()) {
    attentionTreeEncDec.loadCorpus(srcTest, tgtTest, srcParsedTest, testData); 
    for (int i = 0; i< (int)testData.size(); ++i) { // make AttentionTreeEncDec::State* for testData
      testData[i]->state = new AttentionTreeEncDec::State;
      attentionTreeEncDec.makeState(testData[i], testData[i]->state);
    }
    std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
    std::cout << "# of Evaluation Data:\t" << testData.size() << std::endl;
    std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
    std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  } else {}
  
  // Model Loaded...
  attentionTreeEncDec.load(loadModelName);

  Real lossTest = 0., tgtNum = 0.;

#pragma omp parallel for num_threads(attentionTreeEncDec.threadNum) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    Real loss = attentionTreeEncDec.calcLoss(testData[i], testData[i]->state);
#pragma omp critical
    {
      lossTest += loss;
      tgtNum += testData[i]->tgt.size();
    }
  }

  std::cout << "Evaluation Data Perplexity and Loss (/sentence):  " 
	    << exp(lossTest/tgtNum) << ", "
	    << lossTest/testData.size() << "; " 
	    << testData.size() << std::endl;

  static std::unordered_map<int, std::unordered_map<int, Real> > stat; // <src, <trg, Real>>; Real = p(len(trg) | len(src))
  // Load only when translate() called for the first time
attentionTreeEncDec.readStat(stat);

#pragma omp parallel for num_threads(attentionTreeEncDec.threadNum) // ThreadNum
for (int i = 0; i < (int)testData.size(); ++i) {

  auto evalData = testData[i];
  attentionTreeEncDec.translate2(evalData->src, evalData->srcParsed, evalData->srcParsedFlag, evalData->trans,
				 evalData->state, stat, attentionTreeEncDec.beamSize, attentionTreeEncDec.maxGeneNum);
  }

  std::ofstream outputFile;
  std::ostringstream oss;
  std::string parsedMode;
  if (attentionTreeEncDec.parsedTree == AttentionTreeEncDec::PARSED) {
    parsedMode = "PARSED";
  } else { parsedMode = "NONE";}
  oss << attentionTreeEncDec.saveDirName << "Model_AttentionTreeEncDec."
      << "ParsedMode_" << parsedMode
      << ".BlackOut_" << (attentionTreeEncDec.useBlackOut?"true":"false")
      << ".beamSize_" << attentionTreeEncDec.beamSize 
      << ".miniBatchSize_" << attentionTreeEncDec.miniBatchSize
      << ".threadNum_" << attentionTreeEncDec.threadNum
      << ".lrSGD_"<< attentionTreeEncDec.learningRate 
      << ".lrSchedule_"<<  (attentionTreeEncDec.learningRateSchedule?"true":"false")
      << ".vocaThreshold_"<< attentionTreeEncDec.vocaThreshold
      << ".inputFeeding_"<< (attentionTreeEncDec.inputFeeding?"true":"false")
      << ".startIter_"<< startIter
      << ".OutputDev.translate2"; // or OutputTest
  outputFile.open(oss.str(), std::ios::out);

  for (int i=0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    for (auto it = evalData->trans.begin(); it != evalData->trans.end(); ++it) {
      outputFile << attentionTreeEncDec.targetVoc.tokenList[*it]->str << " ";
    }
    outputFile << std::endl;
    // trans
    testData[i]->trans.clear();
  }
}

void AttentionTreeEncDec::save(const std::string& fileName) {
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);  
  
  this->enc.save(ofs);
  this->dec.save(ofs);
  this->encTree.save(ofs);
  this->initDecTree.save(ofs);
  this->softmax.save(ofs);
  this->blackOut.save(ofs);
  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);

  Utils::save(ofs, Wst);
  Utils::save(ofs, bs);
  Utils::save(ofs, Wct);

  Utils::save(ofs, WgeneralTree);

  Utils::save(ofs, WcellSeq);
  Utils::save(ofs, WcellTree);
  Utils::save(ofs, WhSeq);
  Utils::save(ofs, WhTree);
}

void AttentionTreeEncDec::load(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->enc.load(ifs);
  this->dec.load(ifs);
  this->encTree.load(ifs);
  this->initDecTree.load(ifs);
  this->softmax.load(ifs);
  this->blackOut.load(ifs);
  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
  
  Utils::load(ifs, Wst);
  Utils::load(ifs, bs);
  Utils::load(ifs, Wct);

  Utils::load(ifs, WgeneralTree);

  Utils::load(ifs, WcellSeq);
  Utils::load(ifs, WcellTree);
  Utils::load(ifs, WhSeq);
  Utils::load(ifs, WhTree);
}
