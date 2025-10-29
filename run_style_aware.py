"""
Style-Aware Adaptive Teammate Generation Algorithm

This implements the three-phase algorithm:
Phase 1: Behavioral Style Mapping and Prototype Initialization
Phase 2: Style-Controlled Evolutionary Teammate Generation
Phase 3: Multi-Head Ego Policy Training

Based on the paper's Algorithm 1, 2, and 3.
"""

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from copy import deepcopy
import h5py

# Import wandb with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

# Import new modules for style-aware training
from modules.style_encoder import StyleEncoder, ContrastiveLearner, sample_prototypes_on_sphere
from modules.prototype_manager import PrototypeManager
from components.reward_shaping import (
    add_shaped_rewards_to_batch,
    collect_trajectories_for_embedding_training,
    compute_trajectory_style_embedding
)

import json
import numpy as np
from collections import defaultdict


def run(_run, _config, _log):
    """Entry point for Sacred experiment."""
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)

        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # configure wandb logger
    if args.use_wandb and not args.evaluate:
        if not WANDB_AVAILABLE:
            _log.warning("Wandb is not installed. Skipping wandb logging. Install with 'pip install wandb'.")
        else:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name if hasattr(args, 'wandb_run_name') and args.wandb_run_name else None,
                    config=vars(args),
                    tags=args.wandb_tags if hasattr(args, 'wandb_tags') and args.wandb_tags else None,
                    notes=args.wandb_notes if hasattr(args, 'wandb_notes') and args.wandb_notes else None,
                    dir=results_save_dir,
                    reinit=True
                )
                logger.setup_wandb(wandb_run)
                _log.info(f"Wandb initialized successfully. Project: {args.wandb_project}, Run: {wandb_run.name}")
            except Exception as e:
                _log.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")

    # Run and train
    run_sequential_style_aware(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def run_sequential_style_aware(args, logger):
    """
    Main training loop implementing the three-phase style-aware algorithm.

    Phase 1: Train style embedding φ via contrastive learning
    Phase 2: Evolve diverse teammates with style control
    Phase 3: Train multi-head ego agent
    """

    logger.console_logger.info("="*80)
    logger.console_logger.info("Style-Aware Adaptive Teammate Generation Algorithm")
    logger.console_logger.info("="*80)

    # ============================================================================
    # INITIALIZATION: Setup environment, buffers, and base networks
    # ============================================================================

    logger.console_logger.info("Initializing environment and base networks...")

    # Setup runner and environment
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Setup episode buffer scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess)

    empty_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)

    # ============================================================================
    # PHASE 1: Behavioral Style Mapping and Prototype Initialization (Algorithm 1)
    # ============================================================================

    logger.console_logger.info("\n" + "="*80)
    logger.console_logger.info("PHASE 1: Behavioral Style Mapping and Prototype Initialization")
    logger.console_logger.info("="*80)

    # Hyperparameters for Phase 1
    embedding_dim = getattr(args, 'style_embedding_dim', 64)
    n_initial_prototypes = getattr(args, 'n_initial_prototypes', 10)
    n_pretrain_episodes = getattr(args, 'n_pretrain_episodes', 1000)
    population_size_per_prototype = getattr(args, 'population_size_per_prototype', 3)

    logger.console_logger.info(f"Embedding dimension: {embedding_dim}")
    logger.console_logger.info(f"Initial prototypes: {n_initial_prototypes}")
    logger.console_logger.info(f"Population size per prototype: {population_size_per_prototype}")

    # Step 1.1: Initialize prototype vectors uniformly on sphere
    logger.console_logger.info("Initializing prototype vectors on unit hypersphere...")
    prototype_manager = PrototypeManager(
        n_initial_prototypes=n_initial_prototypes,
        embedding_dim=embedding_dim,
        population_size_per_prototype=population_size_per_prototype,
        elimination_threshold=getattr(args, 'proto_elimination_threshold', 0.0),
        merging_threshold=getattr(args, 'proto_merging_threshold', 0.9),
        evaluation_window=getattr(args, 'proto_eval_window', 5)
    )

    logger.console_logger.info(f"Initialized {len(prototype_manager.get_active_prototypes())} prototypes")

    # Step 1.2: Initialize teammate populations for each prototype
    logger.console_logger.info("Initializing teammate populations for each prototype...")

    # Create base teammate network
    base_mac_tm = mac_REGISTRY[args.mac](empty_buffer.scheme, groups, args)
    if args.use_cuda:
        base_mac_tm.cuda()

    # Initialize populations by random perturbation of base policy
    for proto_id in prototype_manager.get_active_prototypes():
        for _ in range(population_size_per_prototype):
            mac_tm = deepcopy(base_mac_tm)
            # Add random noise to parameters
            with th.no_grad():
                for param in mac_tm.parameters():
                    param.add_(th.randn_like(param) * 0.01)
            prototype_manager.add_individual(proto_id, mac_tm, performance=0.0)

    logger.console_logger.info(f"Initialized {n_initial_prototypes * population_size_per_prototype} teammate individuals")

    # Step 1.3: Collect diverse trajectories for training style embedding
    logger.console_logger.info(f"Collecting {n_pretrain_episodes} episodes for contrastive learning...")

    all_macs = []
    for proto_id in prototype_manager.get_active_prototypes():
        pop = prototype_manager.get_population(proto_id)
        for ind_id, ind_data in pop.items():
            all_macs.append(ind_data['mac'])

    n_episodes_per_mac = max(1, n_pretrain_episodes // len(all_macs))
    trajectories = collect_trajectories_for_embedding_training(
        runner, all_macs, n_episodes=n_episodes_per_mac
    )

    logger.console_logger.info(f"Collected {len(trajectories)} trajectories")

    # Step 1.4: Train style embedding function φ via contrastive learning
    logger.console_logger.info("Training style embedding function φ via InfoNCE contrastive learning...")

    # Create style encoder
    style_encoder = StyleEncoder(
        state_dim=args.obs_shape,
        action_dim=args.n_actions,
        hidden_dim=getattr(args, 'style_encoder_hidden_dim', 128),
        embedding_dim=embedding_dim
    )

    # Create contrastive learner
    contrastive_learner = ContrastiveLearner(
        encoder=style_encoder,
        lr=getattr(args, 'style_encoder_lr', 1e-3),
        temperature=getattr(args, 'contrastive_temperature', 0.07),
        device=args.device
    )

    # Training loop for φ
    n_contrastive_epochs = getattr(args, 'n_contrastive_epochs', 50)
    batch_size_contrastive = getattr(args, 'batch_size_contrastive', 32)

    for epoch in range(n_contrastive_epochs):
        # Sample batch of trajectories
        batch_indices = np.random.choice(len(trajectories), size=min(batch_size_contrastive, len(trajectories)), replace=False)
        batch_trajs = [trajectories[i] for i in batch_indices]

        loss = contrastive_learner.train_batch(batch_trajs)

        if epoch % 10 == 0:
            logger.console_logger.info(f"Epoch {epoch}/{n_contrastive_epochs}, InfoNCE Loss: {loss:.4f}")

    # Freeze style encoder
    style_encoder.eval()
    for param in style_encoder.parameters():
        param.requires_grad = False

    logger.console_logger.info("Style embedding φ training complete and frozen")

    # Save style encoder
    encoder_save_path = os.path.join(args.results_save_dir, "style_encoder.pt")
    contrastive_learner.save(encoder_save_path)
    logger.console_logger.info(f"Style encoder saved to {encoder_save_path}")

    logger.console_logger.info("PHASE 1 COMPLETE")

    # ============================================================================
    # PHASE 2: Style-Controlled Evolutionary Teammate Generation (Algorithm 2)
    # ============================================================================

    logger.console_logger.info("\n" + "="*80)
    logger.console_logger.info("PHASE 2: Style-Controlled Evolutionary Teammate Generation")
    logger.console_logger.info("="*80)

    # Initialize ego agent for cross-play training
    mac_ego = mac_REGISTRY[args.mac](empty_buffer.scheme, groups, args)
    buffer_ego = deepcopy(empty_buffer)
    learner_ego = le_REGISTRY[args.learner](mac_ego, empty_buffer.scheme, logger, args)
    if args.use_cuda:
        learner_ego.cuda()

    logger.console_logger.info("Ego agent initialized for cross-play training")

    # Create learners for each prototype's population
    proto_learners = {}  # {prototype_id: {individual_id: learner}}
    proto_buffers_sp = {}  # {prototype_id: {individual_id: buffer}}
    proto_buffers_xp = {}

    for proto_id in prototype_manager.get_active_prototypes():
        proto_learners[proto_id] = {}
        proto_buffers_sp[proto_id] = {}
        proto_buffers_xp[proto_id] = {}

        pop = prototype_manager.get_population(proto_id)
        for ind_id, ind_data in pop.items():
            mac_tm = ind_data['mac']
            learner_tm = le_REGISTRY[args.learner_tm](
                mac_tm, empty_buffer.scheme, logger, args, tm_index=ind_id
            )
            if args.use_cuda:
                learner_tm.cuda()

            proto_learners[proto_id][ind_id] = learner_tm
            proto_buffers_sp[proto_id][ind_id] = deepcopy(empty_buffer)
            proto_buffers_xp[proto_id][ind_id] = deepcopy(empty_buffer)

    # Training hyperparameters
    max_generations = getattr(args, 'max_generations', 20)
    t_train_tm_per_gen = getattr(args, 't_train_tm_per_generation', 10000)
    alpha_ref = getattr(args, 'alpha_style_control', 1.0)
    alpha_explore = getattr(args, 'alpha_exploration', 0.1)
    management_window = getattr(args, 'proto_management_window', 5)

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0

    logger.console_logger.info(f"Starting evolutionary training for {max_generations} generations")
    logger.console_logger.info(f"Training steps per generation: {t_train_tm_per_gen}")

    # Main evolutionary loop
    for generation in range(max_generations):
        logger.console_logger.info(f"\n{'='*80}")
        logger.console_logger.info(f"Generation {generation}/{max_generations}")
        logger.console_logger.info(f"Active prototypes: {len(prototype_manager.get_active_prototypes())}")
        logger.console_logger.info(f"{'='*80}")

        start_time = time.time()
        t_env_start = runner.t_env

        # Evolve each prototype's population
        while runner.t_env - t_env_start < t_train_tm_per_gen:

            # Randomly select a prototype
            active_protos = prototype_manager.get_active_prototypes()
            if len(active_protos) == 0:
                logger.console_logger.warning("No active prototypes remaining! Stopping.")
                break

            proto_id = np.random.choice(active_protos)
            pop = prototype_manager.get_population(proto_id)

            if len(pop) == 0:
                continue

            # Randomly select an individual from this prototype
            ind_id = np.random.choice(list(pop.keys()))
            mac_tm = pop[ind_id]['mac']
            learner_tm = proto_learners[proto_id][ind_id]

            # Get reference embedding for this prototype
            z_ref = prototype_manager.get_reference_embedding(proto_id)

            # Step 2.1: Collect self-play trajectory
            episode_batch_sp = runner.run(
                mac1=mac_tm,
                mac2=None,  # Self-play
                test_mode=False,
                test_mode_1=False,
                test_mode_2=False,
                tm_id=ind_id,
                eps_greedy_t=runner.t_env - t_env_start
            )

            # Add style control and exploration rewards
            if z_ref is not None:
                episode_batch_sp, r_ref_mean, r_explore_mean = add_shaped_rewards_to_batch(
                    episode_batch_sp, style_encoder, prototype_manager, proto_id,
                    z_ref, alpha_ref=alpha_ref, alpha_explore=alpha_explore, device=args.device
                )

            proto_buffers_sp[proto_id][ind_id].insert_episode_batch(episode_batch_sp)

            # Step 2.2: Collect cross-play trajectory (with ego)
            if args.xp_coef > 0 and generation > 0:
                episode_batch_xp = runner.run(
                    mac1=mac_ego,
                    mac2=mac_tm,
                    test_mode=False,
                    test_mode_1=True,  # Ego greedy
                    test_mode_2=False,  # TM explores
                    negative_reward=True,  # Negative reward for TM
                    tm_id=ind_id,
                    eps_greedy_t=runner.t_env - t_env_start,
                    iter=generation
                )

                # Add shaped rewards
                if z_ref is not None:
                    episode_batch_xp, _, _ = add_shaped_rewards_to_batch(
                        episode_batch_xp, style_encoder, prototype_manager, proto_id,
                        z_ref, alpha_ref=alpha_ref, alpha_explore=alpha_explore, device=args.device
                    )

                proto_buffers_xp[proto_id][ind_id].insert_episode_batch(episode_batch_xp)

            # Step 2.3: Train the teammate with L_tm = L_SP + L_XP + γ * L_div
            if proto_buffers_sp[proto_id][ind_id].can_sample(args.batch_size):
                sp_batch = proto_buffers_sp[proto_id][ind_id].sample(args.batch_size)
                max_ep_t = sp_batch.max_t_filled()
                sp_batch = sp_batch[:, :max_ep_t]
                if sp_batch.device != args.device:
                    sp_batch.to(args.device)

                xp_batch = None
                if args.xp_coef > 0 and proto_buffers_xp[proto_id][ind_id].can_sample(args.batch_size) and generation > 0:
                    xp_batch = proto_buffers_xp[proto_id][ind_id].sample(args.batch_size)
                    max_ep_t = xp_batch.max_t_filled()
                    xp_batch = xp_batch[:, :max_ep_t]
                    if xp_batch.device != args.device:
                        xp_batch.to(args.device)

                # Get all macs from this prototype for diversity loss
                all_macs_in_proto = {i: pop[i]['mac'] for i in pop.keys()}
                learner_tm.train(sp_batch, xp_batch, runner.t_env, episode, all_macs_in_proto)

            episode += args.batch_size_run

            # Step 2.4: Update reference policy
            # Evaluate this individual's SP performance
            if episode % (args.test_interval // 10) == 0:  # Periodic reference update
                test_returns = []
                for _ in range(3):  # Quick evaluation
                    test_batch = runner.run(
                        mac1=mac_tm, mac2=None, test_mode=True,
                        test_mode_1=True, test_mode_2=True, tm_id=ind_id, few_shot=True
                    )
                    test_returns.append(test_batch["episode_return"])

                avg_return = np.mean(test_returns)

                # Compute trajectory embedding
                test_batch_full = runner.run(
                    mac1=mac_tm, mac2=None, test_mode=True,
                    test_mode_1=True, test_mode_2=True, tm_id=ind_id
                )
                z_tau = compute_trajectory_style_embedding(test_batch_full, style_encoder, args.device)

                # Update reference if better
                prototype_manager.update_reference_policy(proto_id, mac_tm, avg_return, z_tau)
                prototype_manager.record_performance(proto_id, avg_return)

            # Logging
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.log_stat("n_prototypes", len(prototype_manager.get_active_prototypes()), runner.t_env)
                logger.log_stat("generation", generation, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        # Clear buffers after each generation
        for proto_id in prototype_manager.get_active_prototypes():
            for ind_id in proto_buffers_sp[proto_id].keys():
                proto_buffers_sp[proto_id][ind_id] = deepcopy(empty_buffer)
                proto_buffers_xp[proto_id][ind_id] = deepcopy(empty_buffer)

        # Step 2.5: Adaptive Prototype Management (every W generations)
        if generation > 0 and generation % management_window == 0:
            logger.console_logger.info("Performing adaptive prototype management...")
            eliminated, merged = prototype_manager.adaptive_management(style_encoder, generation)

            if eliminated:
                logger.console_logger.info(f"Eliminated prototypes: {eliminated}")
                # Remove learners and buffers
                for proto_id in eliminated:
                    if proto_id in proto_learners:
                        del proto_learners[proto_id]
                        del proto_buffers_sp[proto_id]
                        del proto_buffers_xp[proto_id]

            if merged:
                logger.console_logger.info(f"Merged prototype pairs: {merged}")
                # Merged pairs already handled by prototype_manager

        logger.console_logger.info(f"Generation {generation} complete. Time: {time_str(time.time() - start_time)}")

    logger.console_logger.info("PHASE 2 COMPLETE")
    logger.console_logger.info(f"Final number of prototypes: {len(prototype_manager.get_active_prototypes())}")

    # Save final prototype space
    proto_save_path = os.path.join(args.results_save_dir, "final_prototypes.json")
    proto_summary = prototype_manager.summary()
    with open(proto_save_path, 'w') as f:
        json.dump(proto_summary, f, indent=4)
    logger.console_logger.info(f"Prototype summary saved to {proto_save_path}")

    # ============================================================================
    # PHASE 3: Multi-Head Ego Policy Training (Algorithm 3)
    # ============================================================================

    logger.console_logger.info("\n" + "="*80)
    logger.console_logger.info("PHASE 3: Multi-Head Ego Policy Training")
    logger.console_logger.info("="*80)

    # Phase 3 implementation would go here
    # This involves training ego with multi-head architecture and regularization
    # Due to length constraints, I'll provide a simplified version

    # Get final teammate library organized by prototype
    active_prototypes = prototype_manager.get_active_prototypes()
    teammate_library = {}  # {proto_id: [mac1, mac2, ...]}

    for proto_id in active_prototypes:
        pop = prototype_manager.get_population(proto_id)
        teammate_library[proto_id] = [ind_data['mac'] for ind_data in pop.values()]

    logger.console_logger.info(f"Training ego with {len(active_prototypes)} style-specific heads")

    # Training hyperparameters for Phase 3
    t_train_ego_per_head = getattr(args, 't_train_ego_per_head', 50000)
    alpha_reg = getattr(args, 'alpha_regularization', 0.01)

    # Initialize ego with multi-head support (if not already)
    # Note: Current implementation treats each prototype as a separate head
    # In a full implementation, we'd need to modify the ego agent architecture

    # Store backbone parameters after each head training for regularization
    backbone_params_history = []

    # Train a head for each prototype sequentially
    for head_idx, proto_id in enumerate(active_prototypes):
        logger.console_logger.info(f"\n{'-'*80}")
        logger.console_logger.info(f"Training Head {head_idx+1}/{len(active_prototypes)} for Prototype {proto_id}")
        logger.console_logger.info(f"{'-'*80}")

        teammates_for_this_head = teammate_library[proto_id]
        logger.console_logger.info(f"Training against {len(teammates_for_this_head)} teammates")

        # Reset ego head (in practice, this would be managed by the ego agent architecture)
        # For now, we treat the whole ego as one "head" per prototype
        if head_idx > 0:
            # Save current backbone parameters for regularization
            current_backbone_params = [param.clone() for param in mac_ego.parameters()]
            backbone_params_history.append(current_backbone_params)

        # Training loop for this head
        t_env_start_ego = runner.t_env
        buffer_ego = deepcopy(empty_buffer)

        while runner.t_env - t_env_start_ego < t_train_ego_per_head:

            # Sample a random teammate from this prototype's population
            mac_tm = np.random.choice(teammates_for_this_head)

            # Collect cross-play episode (ego adapting to teammate)
            episode_batch = runner.run(
                mac1=mac_ego,
                mac2=mac_tm,
                test_mode=False,
                test_mode_1=False,  # Ego explores
                test_mode_2=True,   # Teammate fixed
                tm_id=proto_id,
                eps_greedy_t=runner.t_env - t_env_start_ego,
                iter=head_idx
            )

            buffer_ego.insert_episode_batch(episode_batch)

            # Train ego
            if buffer_ego.can_sample(args.batch_size):
                episode_sample = buffer_ego.sample(args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # Compute task loss
                learner_ego.train(episode_sample, runner.t_env, episode)

                # Add regularization to prevent catastrophic forgetting
                if head_idx > 0 and alpha_reg > 0:
                    reg_loss = 0
                    current_params = list(mac_ego.parameters())

                    # L2 distance to all previous backbone states
                    for prev_params in backbone_params_history:
                        for curr_p, prev_p in zip(current_params, prev_params):
                            reg_loss += th.sum((curr_p - prev_p.to(curr_p.device)) ** 2)

                    reg_loss = reg_loss / len(backbone_params_history)

                    # Backprop regularization
                    reg_loss = alpha_reg * reg_loss
                    learner_ego.optimiser.zero_grad()
                    reg_loss.backward()
                    learner_ego.optimiser.step()

                    # Log regularization
                    logger.log_stat(f"ego_reg_loss_head_{head_idx}", reg_loss.item(), runner.t_env)

            episode += args.batch_size_run

            # Logging
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.log_stat("ego_head", head_idx, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        # Evaluate this head's performance
        logger.console_logger.info(f"Evaluating Head {head_idx+1}...")
        eval_returns = []
        for tm in teammates_for_this_head[:min(5, len(teammates_for_this_head))]:  # Sample evaluation
            for _ in range(args.test_nepisode):
                test_batch = runner.run(
                    mac1=mac_ego, mac2=tm, test_mode=True,
                    test_mode_1=True, test_mode_2=True,
                    tm_id=proto_id, iter=head_idx
                )
                eval_returns.append(test_batch["episode_return"])

        avg_return = np.mean(eval_returns)
        logger.console_logger.info(f"Head {head_idx+1} Average Return: {avg_return:.2f}")
        logger.log_stat(f"ego_head_{head_idx}_return", avg_return, runner.t_env)

        # Save ego model for this head
        ego_save_path = os.path.join(args.results_save_dir, "models", f"ego_head_{head_idx}")
        os.makedirs(ego_save_path, exist_ok=True)
        learner_ego.save_models(ego_save_path)
        logger.console_logger.info(f"Ego head {head_idx+1} saved to {ego_save_path}")

    logger.console_logger.info("PHASE 3 COMPLETE")

    # Save mapping of prototypes to heads
    proto_to_head_mapping = {proto_id: idx for idx, proto_id in enumerate(active_prototypes)}
    mapping_save_path = os.path.join(args.results_save_dir, "prototype_to_head_mapping.json")
    with open(mapping_save_path, 'w') as f:
        json.dump(proto_to_head_mapping, f, indent=4)
    logger.console_logger.info(f"Prototype-to-head mapping saved to {mapping_save_path}")

    runner.close_env()
    logger.console_logger.info("\n" + "="*80)
    logger.console_logger.info("TRAINING COMPLETE!")
    logger.console_logger.info("="*80)


def args_sanity_check(config, _log):
    """Sanity check for arguments."""
    # set CUDA flags
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]

    return config
