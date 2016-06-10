# Tree2Seq: Tree-to-Sequence Attentional Neural Machine Translation
We have proposed a novel syntactic ANMT model, "[Tree-to-Sequence Attentional Neural Machine Translation](http://arxiv.org/abs/1603.06075)" [1].
We extend an original sequence-to-sequence model [2] with the source-side phrase structure. 
Our model has an attention mechanism that enables the decoder to generate a translated word while softly aligning it with source phrases and words.
[Here](http://www.logos.t.u-tokyo.ac.jp/~eriguchi/demo/tree2seq/index.php) is an online demo of Tree2Seq.

## Description
C++ codes of the syntactic Attention-based Neural Machine Translation (ANMT) model.

1. `AttentionTreeEncDec.xpp`: our ANMT model, "Tree-to-Sequence Attentional Neural Machine Translation"
2. `AttentionEncDec.xpp`: Baseline ANMT model [3]
3. `/data/`: Tanaka Corpus (EN-JP) [4]

## Requirement
  * Eigen, a template libary for linear algebra (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
  * N3LP, C++ libaray for neural network-based NLP (<https://github.com/hassyGo/N3LP>)
  * Boost, C++ library for tree structure (<http://www.boost.org/>)
  * Option: Enju, a syntactic parser for English (<http://kmcs.nii.ac.jp/enju/?lang=en>)

## Usage
   1. Modify the paths of `EIGEN_LOCATION`, `SHARE_LOCATION` and `BOOST_LOCATION`. See `Makefile`. 
   2. `$ bash setup.sh`
   3. `$./tree2seq` (Then, training the `AttentionTreeEncDec` model starts.)
   4. Modify `main.cpp` if you want to change the model.

   (!) Attention: I prepare a small corpus of Tanaka corpus. You need over 100,000 parallel corpus.

## Citation
   * [1] Akiko Eriguchi, Kazuma Hashimoto, and Yoshimasa Tsuruoka. 2015. "[Tree-to-Sequence Attentional Neural Machine Translation](http://www.logos.t.u-tokyo.ac.jp/~eriguchi/paper/ACL2016/ACL2016.pdf)". In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016).
   * [2] I. Sutskever, O. Vinyals, and Q. V. Le. 2014. "[Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)". In Proceedings of Advances in Neural Information Processing Systems 27 (NIPS2014).
   * [3] T. Luong, H. Pham, and C. D. Manning. 2015. "[Effective Approaches to Attention-based Neural Machine Translation](http://www.aclweb.org/anthology/D15-1166)". In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP2015).
   * [4] [Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)

## Contact
Thank you for your interests.
If you have any questions and comments, feel free to contact us.
   * eriguchi [.at.] logos.t.u-tokyo.ac.jp
   * hassy [.at.] logos.t.u-tokyo.ac.jp