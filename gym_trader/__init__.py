#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='trader-v0',
    entry_point='gym_trader.envs:TraderEnv',
    timestep_limit=389,
    reward_threshold=1.0,
    nondeterministic = True,
)
