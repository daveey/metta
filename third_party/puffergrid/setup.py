from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

def build_ext(srcs):
    return Extension(
        name="puffergrid." + srcs[0].split('/')[-1].split('.')[0],
        sources=srcs,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        language="c++",
        build_dir='build',
        include_dirs=[numpy.get_include()],
    )

ext_modules = [
    build_ext(["puffergrid/action.pyx"]),
    build_ext(["puffergrid/event.pyx"]),
    build_ext(["puffergrid/grid.cpp"]),
    build_ext(["puffergrid/grid_env.pyx"]),
    build_ext(["puffergrid/grid_object.pyx"]),
    build_ext(["puffergrid/observation_encoder.pyx"]),
    build_ext(["puffergrid/stats_tracker.pyx"]),
]

setup(
    name='puffergrid',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(
        ext_modules,
        language="c++",
        build_dir='build',
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
            "annotation_typing": True,
            "cdivision": True,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "overflowcheck": False,
            "overflowcheck.fold": True,
            "profile": False,
            "linetrace": False,
            "c_string_encoding": "utf-8",
            "c_string_type": "str",
        },
        annotate=True,
    ),
    install_requires=[
        "numpy",
        "cython",
    ],
)
