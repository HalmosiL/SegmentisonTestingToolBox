#!/bin/bash

model=ddcat epsilon=0.01 alpha=0.01 python test_base_line_pgd.py
model=sat epsilon=0.1 alpha=1 python test_base_cosine.py
model=normal epsilon=0.1 alpha=1 python test_base_cosine_combination.py