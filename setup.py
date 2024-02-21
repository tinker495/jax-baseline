from setuptools import find_packages, setup

packages = [package for package in find_packages() if package.startswith("jax_baselines")] + [
    package for package in find_packages() if package.startswith("model_builder")
]
print(packages)
setup(
    name="jax_baselines",
    version="0.0.1",
    packages=packages,
)
