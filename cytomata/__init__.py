# -*- coding: utf-8 -*-
"""
    cytomata
    -----
    Controlling biology with deep reinforcement learning!
    :copyright: (c) 2017 Phuong T. Ho
    :license: MIT, see LICENSE for more details.
"""
from gym.envs.registration import register


register(
    id='cytomatrix-v0',
    entry_point='cytomata.envs:CytomatrixEnv',
)


__name__ = 'cytomata'
__description__ = 'Controlling biology with deep reinforcement learning!'
__version__ = '0.0.1'
__author__ = 'Phuong T. Ho'
__license__ = 'MIT'
__copyright__ = 'Copyright 2017 Phuong T. Ho'
__email__ = 'phuongho43@gmail.com'
__keywords__ = 'cytomata deep-learning reinforcement-learning research cells engineering'
__website__ = 'https://github.com/phuongho43/cytomata'