import unittest
from ast import arg
import tempfile
from envs.griddly.train import make_env_func, register_custom_components, parse_custom_args
from sample_factory.train import run_rl
from sample_factory.cfg.arguments import default_cfg
import sys
from sample_factory.enjoy import enjoy

def train_and_eval(args):
    train_args = [
            "--algo=APPO",
            "--experiment=test_griddly_1",
            "--env=GDY-Forage",
            "--num_workers=5",
            "--device=cpu",
        ] + args

    with tempfile.TemporaryDirectory() as train_dir:
        train_args.append("--train_dir=" + train_dir)

        train_cfg = parse_custom_args(argv=train_args)
        run_rl(train_cfg)
        eval_args = train_args + [
            "--max_num_episodes=10",
            "--no_render"
        ]
        eval_config = parse_custom_args(argv=eval_args, evaluation=True)
        status = enjoy(eval_config)
        return status[1]

class TestGridlyTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_custom_components()

    def test_2x2(self):
        args = [
            "--train_for_env_steps=300000",
            "--forage.num_agents=2",
            "--forage.width_min=4",
            "--forage.width_max=5",
            "--forage.height_min=4",
            "--forage.height_max=5",
            "--forage.energy_per_agent=1",
            "--forage.wall_density=0",
            "--forage.max_env_steps=2",
            "--forage.prediction_error_reward=0"
        ]
        self.assertGreater(train_and_eval(args), 0.99)

    def test_2x2_predict(self):
        args = [
            "--train_for_env_steps=300000",
            "--forage.num_agents=2",
            "--forage.width_min=4",
            "--forage.width_max=5",
            "--forage.height_min=4",
            "--forage.height_max=5",
            "--forage.energy_per_agent=1",
            "--forage.wall_density=0",
            "--forage.max_env_steps=2",
            "--forage.prediction_error_reward=-0.001"
        ]
        self.assertGreater(train_and_eval(args), 0.99)

    def test_8x8(self):
        args = [
            "--train_for_env_steps=80000",
            "--forage.num_agents=2",
            "--forage.width_min=9",
            "--forage.width_max=10",
            "--forage.height_min=9",
            "--forage.height_max=10",
            "--forage.energy_per_agent=1",
            "--forage.wall_density=0",
            "--forage.max_env_steps=16",
        ]
        self.assertGreater(train_and_eval(args), 0.99)

    def test_8x8_w05(self):
        args = [
            "--train_for_env_steps=80000",
            "--forage.num_agents=2",
            "--forage.width_min=9",
            "--forage.width_max=10",
            "--forage.height_min=9",
            "--forage.height_max=10",
            "--forage.energy_per_agent=1",
            "--forage.wall_density=0.05",
            "--forage.max_env_steps=16",
        ]
        self.assertGreater(train_and_eval(args), 0.99)

if __name__ == "__main__":
    unittest.main()
