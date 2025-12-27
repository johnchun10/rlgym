def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rewards import SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward, FaceBallReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
    from rlgym_ppo.util import RLGymV2GymWrapper


    spawn_opponents = True
    team_size = 2
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds), TimeoutCondition(timeout_seconds=game_timeout_seconds))

    # STAGE 1: Early Rewards
    # Focus: Learn to touch ball, move to ball, jump, face ball
    reward_early = CombinedReward(
        (TouchReward(), 50.0),            # Huge reward for touching ball
        (SpeedTowardBallReward(), 5.0),   # Strong guidance to find ball
        (FaceBallReward(), 1.0),          # Encouragement to face ball
        (InAirReward(), 0.15)             # Small bonus to keep jumping alive
    )

    # STAGE 2: Learning to Score
    # Focus: Scoring, hitting towards goal
    reward_mid = CombinedReward(
        (GoalReward(), 10.0),
        (VelocityBallToGoalReward(), 1.0), # Main driving force
        (SpeedTowardBallReward(), 0.1),    # Reduced assistance
        (TouchReward(), 0.1),              # Reduced touch spam
        (InAirReward(), 0.01)
    )

    # SELECT CURRENT STAGE HERE
    reward_fn = reward_early

    obs_builder = DefaultObs(zero_padding=None,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                             boost_coef=1 / 100.0,)

    state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
                                    KickoffMutator())

    renderer = RocketSimVisRenderer()

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=renderer)

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner
    import os

    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    TICK_SKIP = 8
    STEP_TIME = TICK_SKIP / 120

    latest_checkpoint_dir = "data/checkpoints/rlgym-ppo-run/" + str(max(os.listdir("data/checkpoints/rlgym-ppo-run"), key=lambda d: int(d)))


    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty for now.
                      ppo_batch_size=200_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[2048, 2048, 1024, 1024],  # policy network
                      critic_layer_sizes=[2048, 2048, 1024, 1024],  # critic network
                      ts_per_iteration=200_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=400_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=200_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=1e-4,  # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=10_000_000,  # save every 10M steps
                      timestep_limit=1_000_000_000,  # Train for 1B steps
                      log_to_wandb=True, # Set this to True if you want to use Weights & Biases for logging.
                      add_unix_timestamp=False,
                      checkpoint_load_folder=None, # Use latest_checkpoint_dir for saving runs
                      render=True,
                      render_delay = STEP_TIME / 8
                      ) 
    learner.learn()
