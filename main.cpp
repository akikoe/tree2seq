#include "AttentionEncDec.hpp"
#include "AttentionTreeEncDec.hpp"
#include "PreprocessEnju.hpp"
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

  // Preprocess for the dataset
  /* 1. Enju
     Enju parser preprocesses the raw English text.
     You run the following command to get the xml output.
     $ enju < train.rawtext > train.enju
     -- For details on how to use Enju, see the page (http://kmcs.nii.ac.jp/enju/how-to-use?lang=en).
     -- Of course, you can use other parsers (e.g. Stanford Parser), and it requires you to write another code reading a binary tree.

     2. Construct the parallel dataset
     After obtaining the parsed data (i.e. train.enju) with Enju, you have to construct the sequenctial data (i.e. train.en).
     Run the ``PreprocessEnju::extractParsedSentence`` function, and the tokenized source sentence will be extracted from the parsed data.
     -- Enju already tokenized English sentences.
     -- ``PreprocessEnju::extractParsedSentence`` function also checks the sentence length.
     Finally, you should lower the characters in the source sentences at least.
     We consider the final outputs of ``train.enju.token.parsedSuccess.lower``, ``train.enju.parsedSuccess``, and ``train.ja.parsedSucess`` as ``train.en``, ``train.enju`` and ``train.ja``, respectively.
  */
  /*
  const int threshold = 20;
  const std::string originalFile  = "data/train.enju"; // set the path to the parsed text file
  const std::string originalFile2 = "data/train.ja"; // set the path to the tokenized target text file

  const std::string ParsedFileName = originalFile;
  const std::string tgt = originalFile2;
  std::ostringstream fileName2, srcParsedFileName, tgtFileName;
  fileName2 << ParsedFileName << ".token.parsedSuccess";
  srcParsedFileName << ParsedFileName << ".parsedSuccess";
  tgtFileName << tgt << ".parsedSuccess";

  // Extract parsedSuccessPreprocessEnju
  PreprocessEnju::extractParsedSentence(ParsedFileName, fileName2.str(),
					srcParsedFileName.str(), tgt,
					tgtFileName.str(), threshold);
                                        // extract the tokenized sentences
                                        //   and check the sentence length
  */
  return 0;
}
