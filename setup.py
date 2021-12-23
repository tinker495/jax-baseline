from setuptools import setup
setup(
    name='haiku_baselines',
    version='0.0.1',
    packages=['haiku_baselines'],
    install_requires=[
        'requests',
        'mlagents_envs==0.27.0',
        'gym',
        'gym[atari,accept-rom-license]',
        'box2d',
        'box2d-py',
        'jax',
        'einops',
        'dm-haiku',
        'optax',
        'numpy',
        'ray',
        'colabgymrender',
        'cpprb',
        'tensorboardX',
        'importlib; python_version >= "3.5"',
    ]
)