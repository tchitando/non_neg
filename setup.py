from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
    "h5": ["h5py"],
}

setup(
    name="solo-learn",
    packages=find_packages(exclude=["bash_files", "docs", "downstream", "tests", "zoo"]),
    version="1.0.6",
    license="MIT",
    author="solo-learn development team",
    author_email="vturrisi@gmail.com, enrico.fini@gmail.com",
    url="https://github.com/vturrisi/solo-learn",
    keywords=KW,
    install_requires=[
        "einops",
        "tqdm",
        "wandb",
        "scipy",
        "scikit-learn",
        "hydra-core",
    ],
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
