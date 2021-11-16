from distutils.core import setup

setup(
    name="fourbody",
    packages=["fourbody"],
    version="0.2",
    license="MIT",
    description="Phase space parameterisation for four-body decays X -> h1+ h2- h3- h4+",
    long_description="See the projet homepage for details",
    author="Richard Lane",
    author_email="richard.lane6798@gmail.com",
    url="https://github.com/richard-lane/fourbody",
    download_url="https://github.com/richard-lane/fourbody/archive/v_02.tar.gz",
    keywords=[
        "Physics",
        "hadron",
        "Dalitz",
        "phase space",
        "phasespace",
        "root",
        "cern",
    ],
    install_requires=["numpy", "pylorentz"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
