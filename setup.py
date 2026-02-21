from setuptools import setup, find_packages

setup(
    name="mvtec-ad-anomaly-detection",
    version="1.0.0",
    description="PatchCore anomaly detection on MVTec AD benchmark",
    author="DuyNK",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
