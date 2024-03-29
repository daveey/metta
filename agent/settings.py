base = {
   "algo": "APPO",
   "rollout": 256,
   "exploration_loss": "symmetric_kl",
   "exploration_loss_coeff": 0.002,
   "policy_initialization": "orthogonal",
   "learning_rate": "0.0000195",
   "nonlinearity": "elu",
   "batch_size": 16384,
   "decorrelate_experience_max_seconds": 150,
   "value_loss_coeff": 0.976,
}

small_model = base.copy().update({
   "agent_fc_layers": 4,
   "agent_fc_size": 512,
   "rnn_num_layers": 1,
   "rnn_size": 512,
   "rnn_type": "gru",
})

vast_box = base.copy().update({
   "num_workers": 25,
})
