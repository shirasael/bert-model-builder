from setuptools import setup
import os

with open('requirements.txt', 'rb') as f:
    requirements = f.read().splitlines()


setup(
    name="bert_model_builder",
    version="0.1",
    author="Shira Asa-El",
    description="Credit: https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894",
    packages=['bert_model_builder'],
    install_requires=requirements
)
