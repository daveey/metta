from setuptools import setup, find_packages, Command
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop

class BuildGriddlyCommand(Command):
    description = 'Build Griddly'
    user_options = []

    def run(self):
        subprocess.check_call(
            ['./configure.sh'],
            cwd='third_party/griddly',
            shell=True,
        )
        subprocess.check_call(
            ['conan', 'install', 'deps/conanfile.txt', '--profile', 'default',
             '--profile', 'deps/build.profile', '-s', 'build_type=Release',
             '--build', 'missing', '-if', 'build'],
            cwd='third_party/griddly',
        )
        subprocess.check_call(
            ['cmake', '.', '-B', 'build', '-GNinja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake'],
            cwd='third_party/griddly',
        )
        subprocess.check_call(
            ['cmake', '--build', 'build', '--config', 'Release'],
            cwd='third_party/griddly',
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
        "boto3",
        "conan==1.59.0",
        "jmespath",
        "matplotlib",
        "pytest",
        "rich",
        "tabulate",
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
