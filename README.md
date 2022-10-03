# Filipino-Slang
Source code for Filipino slang correction project

### Examples


### Results
We compare the performance of the N-Gram model to other deep learning and baseline models, finding that the N-Grams DLD V1 and V2 perform best on the developed train-test set in terms of accuracy.

<b>Table 1.</b> Test Set Results Across Models
| Model | Accuracy @ 1 | Accuracy @ 3 | Accuracy @ 5 | Min DLD | Mean DLD | Max DLD |
| ----- | ------------ | ------------ | ------------ | ------- | ------- | ------- |
| N-Grams + DLD V1 | <b>0.77</b> |  <b>0.82</b>  |  <b>0.85</b>  |  <b>0.46</b>  | 2.91 | 4.73  |
 | N-Grams + DLD V2     | 0.67 | 0.74 | 0.74 | 1.03 | 2.96 | 4.59  |
 | N-Grams + Likelihood V1 | 0.17 | 0.38 | 0.58 | 1.22 | 3.50 | 5.29  |
 | N-Grams + Likelihood V2 | 0.47 | 0.61 | 0.64 | 1.30 | 3.06 | 4.65  |
| ByT5 (Model Only)                 | 0.31 | 0.42 | 0.49 | 0.98 | 2.71 | 4.38  |
 | ByT5 + $\Pi$-Model   | 0.37 | 0.58 | 0.66 | 0.57 |  <b>2.06</b>  | 3.41  |
 | ByT5 + AE   | 0.04 | 0.04 | 0.04 | 4.28 | 6.69 | 10.2  |
 | Roberta-Tagalog (Model Only)   | 0.00 | 0.00 | 0.00 | 5.79 | 15.3 | 56.7  
 | Roberta-Tagalog + $\Pi$-Model   | 0.00 | 0.00 | 0.00 | 5.69 | 16.5 | 69.2  |
 | Roberta-Tagalog + AE   | 0.00 | 0.00 | 0.00 | 9.44 | 42.8 | 81.7  |
| DLD                  | 0.45 | 0.67 | 0.72 | 0.59 | 2.28 |  <b>3.32</b>   |
 | Google Translate     | 0.44 | -    | -   | -    | -    | -   |   

Upon cross validating, the N-Grams DLD still performs best! We only use V2 since we explain that V1 takes too long to run – hence V2 achieves a better run-time vs accuracy tradeoff.

<b>Table 2.</b> Cross Validated Test Set (k=5) Results Across Models
| Model | Accuracy @ 1 | Accuracy @ 3 | Accuracy @ 5 | Min DLD | Mean DLD | Max DLD |
| ----- | ------------ | ------------ | ------------ | ------- | ------- | ------- |
| N-Grams + DLD V2 | 0.53 ± 0.02 | 0.63 ± 0.04 | 0.65 ± 0.06 | 1.49 ± 0.11 | 2.93 ± 0.07 | 4.18 ± 0.11 |
| N-Grams + Likelihood V2 | 0.35 ± 0.07 | 0.47 ± 0.08 | 0.49 ± 0.07 | 1.69 ± 0.26 | 2.95 ± 0.11 | 4.13 ± 0.16 |
| ByT5 (Model Only) | 0.32 ± 0.06 | 0.52 ± 0.05 | 0.59 ± 0.07 | 0.77 ± 0.15 | 2.31 ± 0.16 | 3.76 ± 0.26 |
| ByT5 + $\Pi$-Model  | 0.40 ± 0.06 | 0.57 ± 0.03 | 0.65 ± 0.03 | 0.53 ± 0.05 | 1.75 ± 0.07 | 2.83 ± 0.12 |
| ByT5 + AE | 0.02 ± 0.03 | 0.02 ± 0.03 | 0.02 ± 0.03 | 4.05 ± 0.41 | 6.33 ± 0.38 | 9.45 ± 0.71 |
| Roberta-Tagalog (Model Only) | 0.0 ± 0.00 | 0.0 ± 0.00 | 0.0 ± 0.00 | 6.06 ± 0.55 | 12.0 ± 2.85 | 46.2 ± 20.0 |
| Roberta-Tagalog + $\Pi$-Model | 0.0 ± 0.00 | 0.0 ± 0.00 | 0.0 ± 0.00 | 6.08 ± 0.56 | 15.3 ± 2.77 | 61.7 ± 17.5 |
| Roberta-Tagalog + AE | 0.0 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 7.38 ± 1.53 | 21.3 ± 6.92 | 54.9 ± 8.10 |

We also experiment with different cutoff values (i.e. number of generated candidates at each step) and compare their time performance and accuracy against cutoffs of 10, 20, 30, 40, 50, 100, 200, 300, and 400.

![alt text](https://github.com/ljyflores/[reponame]/blob/[branch]/image.jpg?raw=true)


 ### Replicating Results
 
 To replicate the results for V2, kindly use this command for Table 1:
 ```python main_algo.py --mode test --use_dld True --max_sub 2 --cutoff 100```
and this for Table 2: 
 ```python main_algo.py --mode eval_cv --use_dld True --max_sub 2 --cutoff 100```
