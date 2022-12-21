#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import glob
import re
from os import path

import setuptools
import torch
from torch.utils.cpp_extension import CppExtension

torch_ver = [int(x) for x in torch.__version__.split('.')[:2]]
assert torch_ver >= [1, 7], 'Requires PyTorch >= 1.7'


with open('damo/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(),
                        re.MULTILINE).group(1)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def get_install_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith('#')]
    return reqs

setuptools.setup(
    name='damo',
    version=version,
    author='basedet team',
    python_requires='>=3.6',
    long_description=long_description,
    install_requires=get_install_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
)
