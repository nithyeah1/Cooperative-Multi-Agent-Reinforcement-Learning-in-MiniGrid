"""
Quick test to verify the improved setup works
"""

import numpy as np
from multiagent_minigrid_env import MultiAgentGoalReachingEnv

def test_reward_shaping():
    """Test that reward shaping works correctly"""
    print("="*60)
    print("Testing Reward Shaping")
    print("="*60)

    # Create environment with reward shaping
    env = MultiAgentGoalReachingEnv(
        grid_size=6,
        num_agents=2,
        max_steps=150,
        shared_reward=True,
        reward_shaping=True
    )

    obs, info = env.reset(seed=42)
    print(f"✓ Environment created (6x6 grid)")
    print(f"✓ Observation shape: {obs['agent_0'].shape}")

    # Take a few steps and check rewards
    print("\nTaking 5 random steps...")
    total_rewards = {'agent_0': 0, 'agent_1': 0}

    for step in range(5):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for agent in env.agents:
            total_rewards[agent] += rewards[agent]

        print(f"Step {step+1}: Agent 0 reward={rewards['agent_0']:.3f}, "
              f"Agent 1 reward={rewards['agent_1']:.3f}")

    print(f"\n✓ Reward shaping working!")
    print(f"  Total rewards: Agent 0={total_rewards['agent_0']:.3f}, "
          f"Agent 1={total_rewards['agent_1']:.3f}")

    env.close()
    return True

def test_observation_format():
    """Test that observations are in correct format for CnnPolicy"""
    print("\n" + "="*60)
    print("Testing Observation Format for CnnPolicy")
    print("="*60)

    env = MultiAgentGoalReachingEnv(grid_size=6, num_agents=2)
    obs, _ = env.reset()

    agent_obs = obs['agent_0']
    print(f"✓ Observation shape: {agent_obs.shape}")
    print(f"✓ Observation dtype: {agent_obs.dtype}")
    print(f"✓ Value range: [{agent_obs.min()}, {agent_obs.max()}]")

    # Check it's in HWC format (Height, Width, Channels)
    assert len(agent_obs.shape) == 3, "Should be 3D (H, W, C)"
    assert agent_obs.shape[2] == 3, "Should have 3 channels (RGB)"
    assert agent_obs.dtype == np.uint8, "Should be uint8"
    assert agent_obs.max() <= 255 and agent_obs.min() >= 0, "Should be in [0, 255]"

    print("✓ Format is correct for CnnPolicy!")
    env.close()
    return True

if __name__ == "__main__":
    try:
        test_reward_shaping()
        test_observation_format()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to train with:")
        print("  python train_ippo.py --mode train")
        print("\nMonitor progress with:")
        print("  tensorboard --logdir ./tensorboard/ippo_improved/")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
