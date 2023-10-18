from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import os


def collect_files(target_dir):
    file_list = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for filename in files:
            file_list.append(os.path.join("..", root, filename))
    return file_list


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        self._run_post_install_tasks()

    def _run_post_install_tasks(self):
        # Post installation tasks:
        try:
            # Install cuda dependencies
            subprocess.check_call(
                ["conda", "install", "--yes", "cudatoolkit==11.8.0", "cudnn"]
            )
        except subprocess.CalledProcessError as e:
            print("Error occurred during post-installation:", e)


class PostDevelopCommand(develop):  # New class
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        self._run_post_install_tasks()

    def _run_post_install_tasks(self):
        # Post installation tasks:
        try:
            # Install cuda dependencies
            subprocess.check_call(
                ["conda", "install", "--yes", "cudatoolkit==11.8.0", "cudnn"]
            )
        except subprocess.CalledProcessError as e:
            print("Error occurred during post-installation:", e)


requirements = [
    "attrs",
    "openai",
    "flask",
    "open3d",
    "trimesh",
    "tensorflow==2.12.0",
    "tensorflow-estimator==2.12.0",
    "tensorflow-io-gcs-filesystem==0.32.0",
    "autolab-core",
    "autolab-perception",
    "visualization",
    "numpy>=1.23.5",
    "scipy",
    "matplotlib",
    "opencv-python",
    "scikit-learn",
    "scikit-image",
    "psutil",
    "gputil",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "scipy>=1.5.0",
    "pyyaml>=5.3.1",
    "pillow",
    "imageio",
    "ninja",
    "gym==0.24.1",
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "rl-games==1.5.2",
    "pyvirtualdisplay",
    "sphinx",
    "myst-parser",
    "sphinx_rtd_theme",
    'einops>=0.6.1',
    'wandb',
    'linformer',
]

packages = find_packages()

package_files = []
package_files = package_files + collect_files("isaacgym/_bindings/linux-x86_64")

setup(
    name="DYNAMO-GRASP",
    version="1.0",
    description="DYNAMO-GRASP - A physics based suction pick point detection method using Isaac GYM simulaton",
    author="Soofiyan Atar, Boling Yang, Markus Grotz",
    author_email="soofiyan2910@gmail.com, bolingy@uw.edu, grotz@uw.edu",
    maintainer="Soofiyan Atar",
    maintainer_email="soofiyan2910@gmail.com",
    description="Project code for running Data Collection using Isaac GYM simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/dynamo-grasp/dynamo-grasp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    packages=packages,
    package_data={"isaacgym": package_files},
    install_requires=requirements,
    extras_require={
        "docs": ["sphinx", "sphinxcontrib-napoleon", "sphinx_rtd_theme"],
    },
    cmdclass={"install": PostInstallCommand, "develop": PostDevelopCommand},
)
