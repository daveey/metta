from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

def build_ext(srcs):
    return Extension(
        name="puffergrid." + srcs[0].split('/')[-1].split('.')[0],
        sources=srcs,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        language="c++",
    )

ext_modules = [
    build_ext(["action.cpp"]),
    build_ext(["event.pyx"]),
    build_ext(["grid.cpp"]),
    build_ext(["grid_env.pyx"]),
    build_ext(["grid_object.pyx"]),
    build_ext(["observation_encoder.pyx"]),
    build_ext(["stats_tracker.pyx"]),
]

setup(
    name='puffergrid',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=[
        "numpy",
        "cython",
    ],
)
