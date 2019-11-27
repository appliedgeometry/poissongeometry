import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PoissonGeometry',
    version='0.1',
    scripts=['poi'] ,
    author="Miguel Evangelista-Alvarado, Jose Ruíz, Pablo Suárez-Serrato",
    author_email="miguel.eva.alv@gmail.com, jcpanta@im.unam.mx, pablo@im.unam.mx",
    license="MIT License"
    description="A Python class for (local) Poisson-Nijenhuis calculus on Poisson manifolds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/appliedgeometry/poisson_geometry",
    packages=setuptools.find_packages(),
    py_modules=['poisson'],
    install_requires=['sympy', 'galgebra'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
