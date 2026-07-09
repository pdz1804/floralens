# Model Card — finetuned_arcface_dinov2_v1

Generated: 2026-07-09T15:58:56Z

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
- Val: {"mrr": 0.9961287693561531, "recall@1": 0.9938875305623472, "precision@1": 0.9938875305623472, "map@1": 0.9938875305623472, "recall@5": 0.9987775061124694, "precision@5": 0.9951100244498776, "map@5": 0.9961559358869871, "recall@10": 0.9987775061124694, "precision@10": 0.9777506112469416, "map@10": 0.995633629878194, "silhouette": 0.9311988353729248, "num_queries": 818, "num_gallery": 1632}
- Test (one-shot): {"mrr": 0.9942950285248573, "recall@1": 0.991442542787286, "precision@1": 0.991442542787286, "map@1": 0.991442542787286, "recall@5": 0.9975550122249389, "precision@5": 0.9938875305623472, "map@5": 0.9946736620483566, "recall@10": 0.9975550122249389, "precision@10": 0.9761613691931526, "map@10": 0.9949125126823345, "silhouette": 0.9360605478286743, "num_queries": 818, "num_gallery": 1632}
- val<->test Recall@5 gap: 0.0012

## Calibration
- Method: isotonic
- ECE (test) before: 0.5860
- ECE (test) after: 0.0001

## Promotion decision
- Decision: **PROMOTE**
- [vs baseline] candidate test Recall@5 (0.9976) within tolerance of baseline (0.9976); val/test gap and ECE within bounds
- [vs active model] candidate test Recall@5 (0.9976) within tolerance of baseline (0.9902); val/test gap and ECE within bounds
