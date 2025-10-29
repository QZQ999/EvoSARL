# Style-Aware Adaptive Teammate Generation

This document describes the implementation of the style-aware adaptive teammate generation algorithm, which extends the original EvoSARL framework with behavioral style prototypes and controlled evolution.

## Algorithm Overview

The algorithm consists of three phases:

### Phase 1: Behavioral Style Mapping and Prototype Initialization
- **Goal**: Learn a style embedding function φ that maps state-action pairs to a d-dimensional unit hypersphere
- **Method**: Contrastive learning (InfoNCE loss) on diverse trajectories
- **Output**: Frozen style encoder φ and K initial prototypes

### Phase 2: Style-Controlled Evolutionary Teammate Generation
- **Goal**: Evolve diverse teammates organized by behavioral style prototypes
- **Key Features**:
  - Style control reward: `r_ref = φ(s,a)^T z_ref` (align with reference policy)
  - Exploration reward: `r_explore = ||φ(s',a') - NN_k(φ(s',a'), H_k)||_2` (novelty)
  - Teammate loss: `L_tm = L_SP + L_XP + γ * L_div`
  - Adaptive prototype management (elimination & merging)

### Phase 3: Multi-Head Ego Policy Training
- **Goal**: Train ego agent with style-specific heads
- **Method**: Sequential head training with regularization to prevent catastrophic forgetting
- **Testing**: Style inference via trajectory embedding similarity

## File Structure

```
EvoSARL/
├── run_style_aware.py              # Main training loop (3 phases)
├── main_style_aware.py             # Entry point with Sacred
├── config/
│   └── style_aware.yaml            # Configuration for style-aware training
├── modules/
│   ├── style_encoder.py            # Style embedding network & contrastive learning
│   └── prototype_manager.py        # Prototype space management
├── components/
│   └── reward_shaping.py           # r_ref and r_explore computation
└── STYLE_AWARE_README.md           # This file
```

## Key Differences from Original EvoSARL

| Aspect | Original | Style-Aware |
|--------|----------|-------------|
| **Structure** | Flat population | Prototype-organized populations |
| **Style** | No explicit representation | φ: (s,a) → ℝ^d hypersphere |
| **Evolution** | Random diversity | Style-controlled + exploration |
| **Ego Training** | Per-teammate heads | Per-prototype heads + regularization |
| **Head Selection** | Few-shot evaluation | Style inference (embedding similarity) |

## Configuration Parameters

Key hyperparameters in `config/style_aware.yaml`:

### Phase 1 (Style Embedding)
```yaml
style_embedding_dim: 64              # Dimension of embedding space
n_initial_prototypes: 10             # K_0, initial prototypes
contrastive_temperature: 0.07        # Temperature for InfoNCE
n_contrastive_epochs: 50             # Training epochs for φ
n_pretrain_episodes: 1000            # Episodes for training data
```

### Phase 2 (Teammate Evolution)
```yaml
max_generations: 20                  # Number of evolutionary generations
t_train_tm_per_generation: 10000     # Training steps per generation
population_size_per_prototype: 3     # Teammates per prototype

alpha_style_control: 1.0             # Weight for r_ref
alpha_exploration: 0.1               # Weight for r_explore
knn_k: 5                             # k for k-NN exploration

proto_elimination_threshold: 0.0     # θ_elim for removing prototypes
proto_merging_threshold: 0.9         # θ_merge for merging prototypes
proto_management_window: 5           # Perform management every W generations
```

### Phase 3 (Ego Training)
```yaml
t_train_ego_per_head: 50000          # Training steps per head
alpha_regularization: 0.01           # Weight for L2 regularization
```

## Usage

### Basic Training

```bash
python main_style_aware.py --env-config=sc2 with env_args.map_name=3m
```

### Custom Configuration

```bash
python main_style_aware.py \
    --env-config=sc2 \
    with env_args.map_name=3m \
    style_embedding_dim=128 \
    n_initial_prototypes=15 \
    max_generations=30 \
    alpha_style_control=2.0
```

### Loading Pretrained Style Encoder

If you have a pretrained style encoder:

```bash
python main_style_aware.py \
    --env-config=sc2 \
    with checkpoint_path=results_style_aware/style_encoder.pt
```

## Output Files

The algorithm saves the following:

```
results_style_aware/
├── style_encoder.pt                 # Trained φ network
├── final_prototypes.json            # Prototype summary
├── prototype_to_head_mapping.json   # Prototype ID → Head index
├── models/
│   ├── ego_head_0/                  # Ego models for each head
│   ├── ego_head_1/
│   └── ...
└── sacred/                          # Sacred experiment logs
```

## Algorithm Pseudocode

### Phase 1: Style Embedding Training
```python
# 1. Initialize K prototypes uniformly on S^{d-1}
Λ = {λ_k} sampled uniformly on hypersphere

# 2. Initialize populations with perturbed policies
For each prototype λ_k:
    P_k = {π_1, ..., π_n} with random perturbations

# 3. Collect diverse trajectories
D = collect_trajectories(all_populations)

# 4. Train φ via InfoNCE contrastive learning
For epoch in epochs:
    Sample batch from D
    Compute InfoNCE loss with temporal positives
    Update φ
Freeze φ
```

### Phase 2: Style-Controlled Evolution
```python
For generation g in max_generations:
    For each training step:
        # Sample prototype and individual
        proto_id = random_choice(active_prototypes)
        ind_id = random_choice(P_{proto_id})

        # Get reference embedding
        z_ref = get_reference_embedding(proto_id)

        # Collect SP trajectory
        τ_sp = collect_episode(π_ind, π_ind)

        # Add shaped rewards
        r' = r + α_ref * φ(s,a)^T z_ref + α_exp * r_explore

        # Collect XP trajectory (with ego)
        τ_xp = collect_episode(π_ego, π_ind)

        # Train with L_tm = L_SP + L_XP + γ * L_div
        train_teammate(τ_sp, τ_xp)

        # Update reference if better
        if performance(π_ind) > performance(π_ref):
            π_ref^{proto_id} = π_ind

    # Adaptive management every W generations
    If g % W == 0:
        eliminate_weak_prototypes()
        merge_similar_prototypes()
```

### Phase 3: Multi-Head Ego Training
```python
backbone_history = []

For each prototype λ_k:
    # Train head k
    For training_step in t_train_ego_per_head:
        # Sample teammate from P_k
        π_tm = random_choice(P_k)

        # Collect XP episode
        τ = collect_episode(π_ego, π_tm)

        # Compute task loss
        L_task = Q_learning_loss(τ)

        # Add regularization
        If k > 0:
            L_reg = Σ ||φ_current - φ_i||^2 / (k-1)
            L_total = L_task + α_reg * L_reg

        # Update ego
        optimize(L_total)

    # Save backbone state
    backbone_history.append(copy(φ_ego))
```

## Testing with Style Inference

After training, select the appropriate head based on style inference:

```python
# 1. Collect short trajectory with test teammate
τ_test = collect_trajectory(π_tm^test, T_infer)

# 2. Compute trajectory embedding
z_test = φ.encode_trajectory(τ_test)

# 3. Find most similar prototype
k* = argmax_k (λ_k^T z_test)

# 4. Activate corresponding head
π_ego = ego_head_{k*}
```

## Implementation Notes

### Current Limitations

1. **Multi-Head Architecture**: The current implementation uses separate training sessions per prototype rather than a true multi-head network. A full implementation would modify the agent architecture to share a backbone with multiple output heads.

2. **Head Expansion**: Unlike the original EvoSARL's dynamic head expansion, this version trains one head per final prototype.

3. **Style Inference**: Currently implemented as post-training analysis. For deployment, you'd need to implement online style inference.

### Future Improvements

1. **True Multi-Head Network**: Implement shared backbone with separate heads in the agent architecture
2. **Online Style Inference**: Real-time prototype detection during episodes
3. **Curriculum Learning**: Progressive prototype difficulty
4. **Meta-Learning**: Learn to quickly adapt to new styles

## Performance Tips

1. **Embedding Dimension**: Start with 64, increase to 128 for complex environments
2. **Initial Prototypes**: Use 10-15 for most tasks, more for highly diverse behaviors
3. **Population Size**: 3-5 per prototype is usually sufficient
4. **Contrastive Training**: More epochs (100+) if style discrimination is poor
5. **Exploration Weight**: Tune α_explore based on environment sparsity

## Citation

If you use this implementation, please cite:
```
[Your paper citation here]
```

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
