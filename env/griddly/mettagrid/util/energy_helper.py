

from env.griddly.builder.action import BehaviorContext

class EnergyHelper():

    def __init__(self, ctx: BehaviorContext, agent) -> None:
        self.agent = agent
        self.ctx = ctx
        self.max_energy = agent.object.cfg.max_energy
        self.energy_reward = agent.object.cfg.energy_reward

    def has_energy(self, energy: int):
        return [ self.agent.energy.gte(energy) ]

    def add(self, amount: int, reason: str):
        cmds = [
            self.agent.energy.add(amount),
            self.ctx.player_var("stats:agent:energy:gained").add(amount),
            self.ctx.player_var(f"stats:agent:energy:gained:{reason}").add(amount),
            self.ctx.cond(self.agent.energy.gt(self.max_energy), [
                self.agent.energy.set(self.max_energy)
            ])
        ]
        if self.energy_reward:
            cmds.append({"reward": amount})

        return cmds

    def use(self, amount: int, reason: str):
        return [
            self.agent.energy.sub(amount),
            self.ctx.player_var("stats:agent:energy:used").add(amount),
            self.ctx.player_var(f"stats:agent:energy:used:{reason}").add(amount),
            self.ctx.cond(self.agent.energy.lt(0), [
                self.agent.energy.set(0)
            ])
        ]
