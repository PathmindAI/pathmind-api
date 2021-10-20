from setuptools import find_packages, setup

setup(
    name="pathmind",
    version="0.2",
    description="Python Simulations ",
    url="https://github.com/PathmindAI/pathmind_api",
    download_url="https://github.com/PathmindAPI/pathmind_api/tarball/0.2",
    author="Max Pumperla",
    author_email="max@pathmind.com",
    install_requires=["pyyaml", "tensorflow", "requests", "prettytable"],
    packages=find_packages(),
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
