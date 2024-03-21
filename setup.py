from setuptools import setup, find_packages, Command
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop

class BuildGriddlyCommand(Command):
    description = 'Build Griddly'
    user_options = []

    def run(self):
        subprocess.check_call(
            ['./build_release.sh'],
            cwd='third_party/griddly',
            shell=True,
        )

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

class DevelopCommand(develop):
    def run(self):
        self.run_command('build_griddly')
        subprocess.check_call(
            ['pip', 'install', "-e", './third_party/griddly/python/'],
        )
        subprocess.check_call(
            ['pip', 'install', "-e", './third_party/sample_factory/'],
        )
        super().run()

setup(
    name='metta',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "awscli",
        "jmespath",
        "matplotlib",
        "pytest",
    ],
    entry_points={
        'console_scripts': [
            # If you want to create any executable scripts in your package
            # For example: 'script_name = module:function'
        ]
    },
    cmdclass={
        'build_griddly': BuildGriddlyCommand,
        'develop': DevelopCommand,
    }
)
