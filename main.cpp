#include "AttentionEncDec.hpp"
#include "AttentionTreeEncDec.hpp"
#include "Utils_tempra28.hpp"

int main(int argc, char** argv){
  const int inputDim = 64;
  const int hiddenDim = 64;
  const Real scale = 0.1;
  const bool useBlackOut = true;
  const int blackOutSampleNum = 200;
  const Real blackOutAlpha = 0.4;
  const bool reversed = false;
  const bool biasOne = true;
  const Real clipThreshold = 3.0;
  const int beamSize = 20;
  const int maxGeneNum = 50;
  const int miniBatchSize = 4;
  const int threadNum = 2;
  const Real learningRate = 0.1;
  const bool learningRateSchedule = false;
  const int srcVocaThreshold = 1;
  const int tgtVocaThreshold = 1;
  const bool inputFeeding = false;
  std::ostringstream saveDirName, loadModelName;
  saveDirName << ""; // TODO: Modify the path
  Eigen::initParallel();
  
  /* Training Data */
  const std::string srcTrain = "data/train.en";
  const std::string tgtTrain = "data/train.ja";
  const std::string srcParsedTrain = "data/train.enju";

  /* Development Data */
  const std::string srcDev = "data/dev.en";
  const std::string tgtDev = "data/dev.ja";
  const std::string srcParsedDev = "data/dev.enju";

  // AttentionTreeEncDec
  AttentionTreeEncDec::demo(srcTrain, tgtTrain, srcParsedTrain, srcDev, tgtDev, srcParsedDev,
			    inputDim, hiddenDim, scale, useBlackOut, blackOutSampleNum, blackOutAlpha,
			    reversed, biasOne, clipThreshold,
			    beamSize, maxGeneNum, miniBatchSize, threadNum, 
			    learningRate, learningRateSchedule, srcVocaThreshold, tgtVocaThreshold,
			    inputFeeding, saveDirName.str());

  /*
  AttentionEncDec::demo(srcTrain, tgtTrain, srcDev, tgtDev,
			inputDim, hiddenDim, scale, useBlackOut, blackOutSampleNum, blackOutAlpha, 
			reversed, biasOne, clipThreshold,
			beamSize, maxGeneNum, miniBatchSize, threadNum, 
			learningRate, learningRateSchedule, srcVocaThreshold, tgtVocaThreshold,
			inputFeeding, saveDirName.str());
  */
  return 0;
}
