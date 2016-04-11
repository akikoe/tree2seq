# Tree2Seq: Tree-to-Sequence Attentional Neural Machine Translation
We have proposed a novel syntactic ANMT model, "[Tree-to-Sequence Attentional Neural Machine Translation Model](http://arxiv.org/abs/1603.06075)" [1].
We extend an original sequence-to-sequence model [2] with the source-side phrase structure. 
Our model has an attention mechanism that enables the decoder to generate a translated word while softly aligning it with source phrases and words.

## Description
C++ codes of the syntactic Attention-based Neural Machine Translation (ANMT) model.

1. `AttentionTreeEncDec.xpp`: our ANMT model, "Tree-to-Sequence Attentional Neural Machine Translation"
2. `AttentionEncDec.xpp`: Baseline ANMT model [3]
3. `/data/`: Tanaka Corpus (EN-JP)

## Requirement
  * Eigen, a template libary for linear algebra (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
  * N3LP, C++ libaray for neural network-based NLP (<https://github.com/hassyGo/N3LP>)
  * Boost, C++ library for tree structure (<http://www.boost.org/>)
  * Option: Enju, a syntactic parser for English (<http://kmcs.nii.ac.jp/enju/?lang=en>)

## Usage
   1. Modify the paths of `EIGEN_LOCATION`, `SHARE_LOCATION` and `BOOST_LOCATION`. See `Makefile`. 
   2. `$ make`
   3. `$./anmt` (Then, training the AttentionTreeEncDec model starts.)
   4. Modify `main.cpp` if you want to change the model.

## Citaion
   * [1] Akiko Eriguchi, Kazuma Hashimoto, and Yoshimasa Tsuruoka. 2015. "[Tree-to-Sequence Attentional Neural Machine Translation](http://arxiv.org/abs/1603.06075)". arXiv cs.CL 1603.06075.
   * [2] [Sutskever et al., 2014](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
   * [3] [Luong et al., 2015](http://www.aclweb.org/anthology/D15-1166)
   * [4] [Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)
