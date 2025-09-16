#  Application of VQShape
What is this?
A streamlined pipeline for detecting “spark” anomalies in GEM1h current time-series using VQShape

## Acknowledgments

Senior’s contribution: This repo This code base is developed based on the code base of Yunshi Wen. Many thanks for the guidance and sharing.

Upstream project: Built on VQShape (see archive/VQSHAPE_README.md).Upstream license:/VQShape/LICENSE
More info, please refer to https://github.com/YunshiWen/VQShape


## Defaults

Slicing: patch_size=30, window_size=5, stride=5

Labels (AD): sliding-mean relative deviation with suppression gap
(defaults ratio=0.005, gap=5)

AP (precursor): shift AD labels by +1 step (--label_mode precursor)

VQ input: linear resample to seq_len=512; features = code usage histogram
(optional --use-prob normalization, --use-tfidf IDF weighting)

Evaluation: pick the best F1 on validation via threshold sweep; report Test
P/R/F1@best_thr, ROC-AUC, PR-AUC.



