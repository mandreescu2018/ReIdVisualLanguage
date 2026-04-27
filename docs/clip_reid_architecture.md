# CLIP-ReID Integration — Architectural Recommendations

CLIP-ReID has two distinct training stages, which is the central design challenge. The current codebase assumes a single flat training run — everything from `train.py` is wired once and handed to `ImageFeatureTrainer.train()`. Multi-stage training breaks that assumption at several levels.

---

## 1. Stage Orchestration (highest priority)

`train.py` builds optimizer, scheduler, and model once. For CLIP-ReID you need **per-stage** variants of all three.

Recommended pattern: a `MultiStageTrainer` that holds a list of `StageConfig` objects, each declaring its own frozen modules, optimizer factory, scheduler, and epoch budget. The orchestrator calls the existing `ImageFeatureTrainer` per stage after reconfiguring the model's `requires_grad` graph.

```python
stages = [
    StageConfig(name="prompt_learning", epochs=30,
                trainable_modules=["text_prompt_learner"],
                frozen_modules=["base", "bottleneck"]),
    StageConfig(name="image_finetuning", epochs=60,
                trainable_modules=["base", "bottleneck", "classifier"],
                frozen_modules=["text_prompt_learner"]),
]
```

Add a `set_trainable(module, flag: bool)` utility to `models/` — it's a 5-line function but you'll call it in several places. Bugs here are subtle: accidentally training "frozen" params is a common source of wasted runs.

---

## 2. Model Architecture Changes

Split responsibilities clearly across three modules:

| Module | Responsibility |
|--------|----------------|
| `CLIPImageEncoder` | Wraps CLIP's ViT, adds SIE camera/view embedding |
| `TextPromptLearner` | Learnable context tokens (`nn.Embedding`) + class token embeddings. Follows CoOp design |
| `CLIPReIDModel` | Orchestrates both, exposes `image_features` and `text_features` |

The key insight: in stage 2, the **text prototypes replace the Linear classifier head**. The current `build_transformer` has `self.classifier = nn.Linear(in_planes, num_classes)` — in CLIP-ReID, this weight matrix is replaced by the frozen text embeddings from stage 1. The `_init_classifier_layers` abstraction is close to right but needs a `from_text_prototypes(embeddings)` factory path.

`ModelInput` / `ModelOutput` dataclasses in `model_io.py` need text fields:

```python
@dataclass
class ModelOutput:
    logits: ...
    features: ...
    text_features: torch.Tensor = None   # for stage 1 contrastive loss
```

---

## 3. Loss Architecture

Stage 1 needs an **image-text contrastive loss** (InfoNCE per identity class). `ComposedLosses` in `losses/` will need a `CLIPContrastiveLoss` that takes the image-text similarity matrix and class labels. Stage 2 reuses the existing triplet + ID loss, but `ID_loss` should accept a weight matrix (the frozen text prototypes) rather than always instantiating `nn.Linear`.

The `w_id * ID + w_metric * Triplet` formula stays the same — only the classifier weights change source.

---

## 4. Checkpoint Strategy

The stage boundary is a natural checkpoint moment. Save stage 1's output as `stage1_prompts.pth` containing only `text_prompt_learner.state_dict()`. Stage 2's entry point should accept `--stage1_ckpt` and load it before freezing. This lets you rerun stage 2 with different hyperparameters without repeating stage 1.

The existing `ModelLoader` / `save_model` in `trainer_base.py` should be extended to handle partial state dicts (loading only prompt weights into a full model).

---

## 5. Config Structure

Add a `CLIP` section to the YAML system (alongside `MODEL`, `SOLVER`, `LOSS`, etc.):

```yaml
CLIP:
  MODEL_NAME: "ViT-B/16"
  N_CTX: 4                    # number of learnable context tokens
  CLASS_TOKEN_POSITION: "end"
  STAGE1_EPOCHS: 30
  STAGE2_EPOCHS: 60
  STAGE1_LR: 0.002
```

Keeping stages in config (not hardcoded in the trainer) means the epoch ratio can be tuned without code changes.

---

## Key Tradeoffs to Decide Before Implementing

**SIE compatibility**: `TransReID` has SIE baked in for camera/view conditioning. CLIP's pretrained ViT doesn't. Port SIE as an additive embedding on top of CLIP's patch embeddings — same mechanism, different initialization point. Apply this in stage 2 only.

**JPM + CLIP**: `build_transformer_local` with its 4 local branches adds significant complexity on top of the CLIP backbone. Implement base CLIP-ReID (global branch only) first, and layer JPM on after a working baseline is confirmed.

**Memory**: Both encoders must coexist in stage 1. CLIP ViT-B/16 image + text encoder is ~360MB total — fine for a single GPU, but factor it into your batch size budget.

---

## Suggested Implementation Order

Given the current `engine/` and `models/` structure, the lowest-risk path:

1. `models/backbones/clip_vit.py` — thin wrapper around `open_clip` or `clip` exposing the same interface as `TransReID`
2. `models/text_prompt_learner.py`
3. Extend `TrainerConfig` with `stage: int` and add `MultiStageTrainer` to `engine/`
4. `losses/clip_contrastive_loss.py`
5. `configurations/Market/clip_reid.yml` with the `CLIP` section

`train.py` barely needs to change — it just instantiates `MultiStageTrainer` instead of `ImageFeatureTrainer`.
