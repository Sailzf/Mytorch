from setuptools import setup, find_packages
import os

setup(
    name="mytorch",
    version="0.1.0",
    author="sail",
    author_email="sailzero@foxmail.com",
    description="A PyTorch-like deep learning framework implementation",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/sail/mytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        # "numpy>=1.19.0",
        # "cupy-cuda11x>=11.0.0",  # 根据你的CUDA版本调整
        # "torchvision",  # 用于加载MNIST数据集
        # "matplotlib",   # 用于绘图
        # "tqdm",        # 用于进度条
        # "nvtx",        # 用于NVIDIA性能分析
        # "scikit-learn", # 用于示例代码
    ],
    extras_require={
        "dev": [
            # "pytest>=6.0",
            # "pytest-cov",
            # # "flake8",
            # # "black",
            # # "isort",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "mytorch-train=examples.mnist_linear:main",  # 可选：添加命令行工具
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)