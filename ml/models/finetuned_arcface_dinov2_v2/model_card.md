# Model Card — finetuned_arcface_dinov2_v2

Generated: 2026-07-09T18:14:58Z

## Data
- Dataset: oxford-102-flowers (hash `1d26d1c1634cd3890ecaa68cd2748df779e6d128`)
- Classes: 102
- Embedded splits: {'gallery': 1632, 'test': 818, 'val': 818} (total 3268)

## Backbone
- dinov2_vitl14 (frozen, input dim 1024)

## Training
- Method: arcface
- Hyperparameters: `{"loss": "arcface", "lr": 0.001, "head_hidden_dim": 256, "output_dim": 256, "margin": 0.3, "scale": 30.0, "batch_size": 128, "max_epochs": 40, "early_stop_patience": 5, "early_stop_min_delta": 0.001, "seed": 42, "weight_decay": 0.0001}`
- Seed: 42

## Evaluation (retrieval protocol, PRD §14.4)
- Val: {"mrr": 0.9965362673186634, "recall@1": 0.9951100244498777, "precision@1": 0.9951100244498777, "map@1": 0.9951100244498777, "recall@5": 0.9987775061124694, "precision@5": 0.9953545232273838, "map@5": 0.995948791089378, "recall@10": 0.9987775061124694, "precision@10": 0.9775061124694354, "map@10": 0.9960828218818091, "silhouette": 0.9358816146850586, "num_queries": 818, "num_gallery": 1632}
- Test (one-shot): {"mrr": 0.9947025264873676, "recall@1": 0.9926650366748166, "precision@1": 0.9926650366748166, "map@1": 0.9926650366748166, "recall@5": 0.9975550122249389, "precision@5": 0.9931540342298288, "map@5": 0.9945242461287694, "recall@10": 0.9975550122249389, "precision@10": 0.9760391198043993, "map@10": 0.994784249387521, "silhouette": 0.9408586025238037, "num_queries": 818, "num_gallery": 1632}
- val<->test Recall@5 gap: 0.0012

## Calibration
- Method: isotonic
- ECE (test) before: 0.5893
- ECE (test) after: 0.0000

## Promotion decision
- Decision: **PROMOTE**
- [vs baseline] candidate test Recall@5 (0.9976) within tolerance of baseline (0.9976); val/test gap and ECE within bounds
- [vs active model] candidate test Recall@5 (0.9976) within tolerance of baseline (0.9976); val/test gap and ECE within bounds
