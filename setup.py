from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vector-db",
    version="0.1.0",
    author="Hong",
    description="A production-grade vector database with HNSW indexing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vector-db",
    packages=find_packages(),
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
