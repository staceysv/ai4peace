
# @title Imports

from collections.abc import Sequence
import random
from typing import Any, Dict, List

from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.contrib import language_models as language_model_utils
from concordia.contrib.components.game_master import marketplace
from concordia.environment.engines import simultaneous
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentence_transformers
import os


# @title Language Model Selection: provide key or select DISABLE_LANGUAGE_MODEL

# By default this colab uses models via an external API so you must provide an
# API key. TogetherAI offers open weights models from all sources.

API_KEY = os.environ['OPENAI_API_KEY']
# See concordia/language_model/utils.py
API_TYPE = 'openai'  # e.g. 'together_ai' or 'openai'.
MODEL_NAME = (  # for API_TYPE = 'together_ai', we recommend MODEL_NAME = 'google/gemma-3-27b-it'
    'gpt-5'
)
# To debug without spending money on API calls, set DISABLE_LANGUAGE_MODEL=True
DISABLE_LANGUAGE_MODEL = False

# @title Use the selected language model

# Note that it is also possible to use local models or other API models,
# simply replace this cell with the correct initialization for the model
# you want to use.

if not DISABLE_LANGUAGE_MODEL and not API_KEY:
  raise ValueError('API_KEY is required.')

model = language_model_utils.language_model_setup(
    api_type=API_TYPE,
    model_name=MODEL_NAME,
    api_key=API_KEY,
    disable_language_model=DISABLE_LANGUAGE_MODEL,
)

# @title Setup sentence encoder

if DISABLE_LANGUAGE_MODEL:
  embedder = lambda _: np.ones(3)
else:
  st_model = sentence_transformers.SentenceTransformer(
      'sentence-transformers/all-mpnet-base-v2'
  )
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

test = model.sample_text(
    'Is societal and technological progress like getting a clearer picture of '
    'something true and deep?'
)
print(test)

# @title Load prefabs from packages to make the specific palette to use here.

prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}

# @title Domain objects for marketplace simulations

Good = marketplace.Good
MarketplaceAgent = marketplace.MarketplaceAgent
MarketPlace = marketplace.MarketPlace


def make_goods() -> List[Good]:
  categories = ["Food", "Clothing", "Accessories", "Gadgets"]
  qualities = ["Low", "Mid", "High"]
  return [Good(c, q, c + "_" + q) for c in categories for q in qualities]


def make_agents(
    n: int, goods: Sequence[Good], names: Sequence[str], seed: int = 123
) -> List[MarketplaceAgent]:
  rng = random.Random(seed)
  agents: List[MarketplaceAgent] = []
  for i in range(n):
    role = "producer" if i < n // 2 else "consumer"
    inv_good = rng.choice(goods)
    inventory = {inv_good.id: rng.randint(5, 15)} if role == "producer" else {}
    agents.append(
        MarketplaceAgent(
            name=names[i],
            role=role,
            cash=50.0,
            inventory=inventory,
            queue=[],
        )
    )
  return agents

# 4 Actor Simulation

goods = make_goods()
names = ['Alex', 'Nicole', 'Jeremy', 'Megan']
agents = make_agents(4, goods, names)

component_kwargs = {
    'components': [
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    ],
    'agents': agents,
    'goods': goods,
}

prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}

PLAYER_ONE = names[0]
PLAYER_TWO = names[1]
PLAYER_THREE = names[2]
PLAYER_FOUR = names[3]

instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': PLAYER_ONE,
            'goal': (
                'Your goal is to sell your stock of Food_Low for a profit. Your'
                ' cost to produce each unit is $2.00. You must try to sell for'
                ' more than $2.00 to be profitable. You will not accept a price'
                ' lower than your cost.'
            ),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': PLAYER_TWO,
            'goal': (
                'Your goal is to sell your stock of Food_Low for a profit. You'
                ' are a very efficient producer, so your cost for each unit is'
                ' only 1.50, but you can'
                ' afford to undercut less efficient sellers.'
            ),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': PLAYER_THREE,
            'goal': (
                'Your goal is to acquire Food_Low for your own use. You value'
                ' it highly, so you are willing to pay up to $4.50 per unit,'
                ' but you will try to get it for as cheap as possible. You will'
                ' not bid higher than $4.50.'
            ),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': PLAYER_FOUR,
            'goal': (
                'Your goal is to acquire Food_Low. It is useful, but not'
                ' essential. You are willing to pay up to $3.75 per unit, but'
                ' you would prefer to pay much less. You will not bid higher'
                ' than $3.75.'
            ),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='marketplace__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'MarketplaceGM',
            'experiment_component_class': MarketPlace,
            'experiment_component_init_kwargs': component_kwargs,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,
        params={
            'name': 'initial setup rules',
            'next_game_master_name': 'MarketplaceGM',
            'shared_memories': [
                (
                    'There is a small town of Smallville where'
                    f' {PLAYER_ONE} and {PLAYER_TWO} grew up.'
                ),
            ],
        },
    ),
]

default_premise = """You are in a marketplace that buys and sells goods
"""

config = prefab_lib.Config(
    default_premise=default_premise,
    default_max_steps=5,
    prefabs=prefabs,
    instances=instances,
)

engine = simultaneous.Simultaneous()

#  Initialize the simulation
simul_sim = simulation.Simulation(
    config=config, model=model, embedder=embedder, engine=engine
)


# @title Run the simulation
results_log = simul_sim.play()


# @title Display the log
display.HTML(results_log)