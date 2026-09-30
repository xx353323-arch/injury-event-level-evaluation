# Data flow and data access by method (SoccerMon, unpurged event-grouped split of the ablation, ranges over 10 seeds)

## Table A. Data flow for each transfer task

| Target | Source labelled windows (N / positives) | Target support labelled (positives / negatives) | Target unlabelled (U_t) | Target test (positives / negatives) |
|---|---|---|---|---|
| TeamA-2020 | 26019 / 104 | 87–218 / 348–872 | support training subset, 290–728 (test excluded) | 209–340 / 8018–8490 |
| TeamA-2021 | 25884 / 469 | 14–20 / 56–80 | support training subset, 48–68 (test excluded) | 42–48 / 9470–9502 |
| TeamB-2020 | 27431 / 496 | 7–19 / 28–76 | support training subset, 24–64 (test excluded) | 16–28 / 7982–8031 |

## Table B. Data components accessible to each method

| Method | Source labelled | Target support labelled | Target unlabelled | Touches test features | Evaluation regime |
|---|---|---|---|---|---|
| GBT | all | all support, concatenated with source | none | no | inductive |
| LSTM | all | as above | none | no | inductive |
| GAT | all | as above | none | no | inductive |
| Transformer | all | as above | none | no | inductive |
| ProtoNet | all, episodic | support forms class prototypes | none | no | inductive |
| DANN | all | all support | support features | no | inductive |
| LMVG-TCA | all, episodic | support training subset for MMD and fine-tuning | support training subset (U_t, features only) | no | inductive |

All methods are compared on the same support budget, the same seeds and the same event-level split.
