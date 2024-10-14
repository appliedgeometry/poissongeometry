import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='poissongeometry',
    version='1.1.1',
    author="Miguel Evangelista-Alvarado, José C. Ruíz Pantaleón, Pablo Suárez-Serrato",
    author_email="miguel.eva.alv@gmail.com, jcpanta@im.unam.mx, pablo@im.unam.mx",
    license="MIT",
    description="A Python module for (local) Poisson-Nijenhuis calculus on Poisson manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/appliedgeometry/poisson_geometry",
    packages=setuptools.find_packages(),
    install_requires=['sympy', 'numpy'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
)
