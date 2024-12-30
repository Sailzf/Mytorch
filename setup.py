from setuptools import setup, find_packages

setup(
    name='mytorch',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    description='A simple torch-like library',
    author='sail',
    author_email='sailzero@foxmail.com',
    url='https://github.com/Sailzf/Mytorch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.1',
    # 不指定install_requires以避免下载依赖
)