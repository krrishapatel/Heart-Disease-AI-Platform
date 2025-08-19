#!/usr/bin/env python3
"""
Setup script for Heart Disease AI Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_github.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Heart Disease AI Platform - Advanced cardiac risk assessment system"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements_github.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="heart-disease-ai-platform",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced cardiac risk assessment system with explainable AI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/heart-disease-ai-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "heart-disease-ai=heart_disease_ai_platform:main",
            "heart-disease-demo=demo_heart_disease:main",
            "heart-disease-test=test_platform:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.data", "*.names", "*.txt", "*.md"],
    },
    keywords=[
        "heart disease",
        "cardiac risk",
        "machine learning",
        "AI",
        "healthcare",
        "medical",
        "SHAP",
        "LIME",
        "explainable AI",
        "cardiovascular",
        "risk assessment",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/heart-disease-ai-platform/issues",
        "Source": "https://github.com/yourusername/heart-disease-ai-platform",
        "Documentation": "https://github.com/yourusername/heart-disease-ai-platform#readme",
    },
)
