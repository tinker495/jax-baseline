from setuptools import setup
setup(
    name='haiku_baselines',
    version='0.0.1',
    packages=['haiku_baselines'],
    install_requires=[
        'requests',
        'mlagents_envs==0.27.0',
        'gym',
        'box2d',
        'box2d-py',
        'git://github.com/kenjyoung/MinAtar.git#egg=MinAtar',
        'dm-haiku',
        'optax',
        'numpy',
        'cpprb',
        'tensorboardX',
        'importlib; python_version >= "3.5"',
    ],
    #dependency_links=['https://github.com/kenjyoung/MinAtar#egg=package-1.0']
)