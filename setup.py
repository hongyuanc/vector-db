import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Define Cython extensions for performance-critical modules
extensions = [
    Extension(
        "src.utils.distance_cy",
        ["src/utils/distance_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=["-O3"],
    ),
    Extension(
        "src.index.hnsw_core",
        ["src/index/hnsw_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-std=c++11"],
        extra_link_args=["-O3"],
        language="c++",  # Use C++ for STL containers
    ),
    Extension(
        "src.index.hnsw_cpp",
        ["src/index/hnsw_cpp.pyx", "src/index/hnsw_cpp_core.cpp"],
        include_dirs=[np.get_include(), "src/index"],
        extra_compile_args=["-O3", "-march=native", "-std=c++11"],
        extra_link_args=["-O3"],
        language="c++",
    ),
]

setup(
    name="vector-db",
    version="0.1.0",
    author="Hong",
    description="A production-grade vector database with HNSW indexing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vector-db",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,  # Disable bounds checking for speed
            "wraparound": False,   # Disable negative indexing for speed
            "cdivision": True,     # Use C division semantics
            "initializedcheck": False,  # Disable memoryview initialization check
        },
        annotate=True,  # Generate HTML annotation files for optimization analysis
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "numba>=0.58.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "sqlalchemy>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.88.0",
            "black>=23.10.0",
            "mypy>=1.6.0",
            "ruff>=0.1.0",
            "pre-commit>=3.5.0",
        ],
        "benchmark": [
            "locust>=2.17.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.0",
            "matplotlib>=3.8.0",
            "pandas>=2.1.0",
            "seaborn>=0.13.0",
        ],
    },
)
