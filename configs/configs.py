# Environments

env_simple = [
    "--env_wall_density=0.01",
    "--env_altar:cost=100",
    "--env_cost:shield=0",
]

env_a5_25x25 = [
    *env_simple,
    "--env_num_agents=5",
    "--env_width=25",
    "--env_height=25",

    "--env_num_altars=1",
    "--env_num_chargers=3",
    "--env_num_generators=15",
]

env_a20_40x40 = [
    *env_simple,
    "--env_num_agents=20",
    "--env_width=40",
    "--env_height=40",

    "--env_num_altars=5",
    "--env_num_chargers=10",
    "--env_num_generators=50",
]

env_a100_100x100 = [
    *env_simple,
    "--env_num_agents=100",
    "--env_width=100",
    "--env_height=100",
    "--env_tile_size=16",

    "--env_num_altars=20",
    "--env_num_chargers=30",
    "--env_num_generators=200",
]


# Agents

rnn = [
    "--rnn_num_layers=1",
    "--rnn_size=512",
    "--rnn_type=gru",
]

obj_embed_agent = [
    *rnn,
    "--agent=object_embedding_agent",
    "--agent_fc_layers=3",
    "--agent_fc_size=512",
    "--agent_embedding_size=64",
    "--agent_embedding_layers=3",
]

obj_attn_agent = [
    *rnn,
    "--agent=object_attn_agent",
    "--agent_attention_combo=mean",
    "--agent_fc_layers=3",
    "--agent_fc_size=512",
]

sandbox_agent = [
    *obj_embed_agent,
    "--agent_fc_layers=1",
    "--agent_fc_size=512",
]


# Training

training = [
    "--normalize_input=False",
    "--aux_loss_coef=0",
    "--recurrence=256",
    "--rollout=256",
    "--value_loss_coeff=0.976",
    "--exploration_loss=symmetric_kl",
    "--exploration_loss_coeff=0.002",
    "--policy_initialization=orthogonal",
    "--learning_rate=0.0001",
    "--max_policy_lag=50",
    "--nonlinearity=elu",
    "--load_checkpoint_kind=latest",
]

prod_training = [
    *training,
    "--with_wandb=True",
    "--wandb_user=platypus",
    "--wandb_project=sample_factory",
    "--batch_size=16384",
    "--decorrelate_experience_max_seconds=150",
]

sandbox_training = [
    *training,
    "--save_every_sec=60",
    "--num_workers=8",
    "--decorrelate_experience_max_seconds=0",
    "--batch_size=1024",
]

# Evaluation

evaluation = [
    "--device=cpu",
    "--fps=6",
    "--max_num_frames=1000",
    "--eval_env_frameskip=1",
    "--load_checkpoint_kind=latest",
    "--train_dir=./train_dir/",
]
