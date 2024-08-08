from sympy import li
import build
from setuptools import Extension, setup, find_packages, Command
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop
from Cython.Build import cythonize
import numpy
import os

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
        os.makedirs("build", exist_ok=True)
        # self.run_command('build_griddly')
        # subprocess.check_call(
        #     ['pip', 'install', "-e", './third_party/griddly/python/'],
        # )
        # subprocess.check_call(
        #     ['pip', 'install', "-e", './third_party/meltingpot/'],
        # ),
        subprocess.check_call(
            ['pip', 'install', "-e", './third_party/sample_factory/'],
        )
        subprocess.check_call(
            ['pip', 'install', "-e", './third_party/pufferlib/'],
        )
        super().run()

def build_ext(srcs, module_name=None):
    subprocess.check_call(['python', 'setup.py', 'build_ext', '--inplace'], cwd='third_party/puffergrid')

    if module_name is None:
        module_name = srcs[0].replace("/", ".").replace(".pyx", "").replace(".cpp", "")
    return Extension(
        module_name,
        srcs,
        # include_dirs=["third_party/puffergrid"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )

ext_modules = [
    build_ext(["env/mettagrid/objects.pyx"]),
    build_ext(["env/mettagrid/actions.pyx"]),
    build_ext(["env/mettagrid/mettagrid.pyx"], "env.mettagrid.mettagrid_c"),
]

debug = os.getenv('DEBUG', '0') == '1'
annotate = os.getenv('ANNOTATE', '0') == '1'

build_dir = 'build'
if debug:
    build_dir = 'build_debug'

os.makedirs(build_dir, exist_ok=True)

setup(
    name='metta',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "boto3",
        # "chex",
        # "conan==1.59.0",
        "hydra-core",
        "jmespath",
        "matplotlib",
        # "numpy==2.0.0",
        "pettingzoo",
        "pynvml",
        "pytest",
        "PyYAML",
        "raylib",
        "rich",
        "scipy",
        "tabulate",
        "tensordict",
        "torchrl",
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
    },
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(
        ext_modules,
        build_dir='build',
        compiler_directives={
            "language_level": "3",
            "embedsignature": debug,
            "annotation_typing": debug,
            "cdivision": debug,
            "boundscheck": debug,
            "wraparound": debug,
            "initializedcheck": debug,
            "nonecheck": debug,
            "overflowcheck": debug,
            "overflowcheck.fold": debug,
            "profile": debug,
            "linetrace": debug,
            "c_string_encoding": "utf-8",
            "c_string_type": "str",
        },
        annotate=debug or annotate,
    ),
)
