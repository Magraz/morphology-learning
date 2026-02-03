import torch
from algorithms.ippo.trainer import IPPOTrainer


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment configuration
    env_config = {
        "render_mode": "human",  # Set to "human" for visual training
        "n_agents": 10,
    }

    # PPO configuration
    ppo_config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    }

    # Create trainer
    trainer = IPPOTrainer(env_config, ppo_config, device)

    # Save trained agents
    trainer.load_agents("/home/magraz/research/src/trained_ppo_agents.pth")

    # Test trained agents with rendering
    print("\nTesting trained agents...")
    trainer.render_episode()
    trainer.env.close()


if __name__ == "__main__":
    main()
