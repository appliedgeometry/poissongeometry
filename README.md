# Poisson Geometry
Is this a Python class to make symbolic calculus in Poisson Geometry, some of its functions are:

 - [x] Get Poisson structures from Flaska-Ratio formula
 - [x] Calculate Schouten Bracket
 - [x] Calculate Poisson Bracket

## Starting 🚀
#### You only have an interest in trying:
 * __On the cloud__
   __TODO__ add link to codelab
 * __On local machine__
   * Clone this repository in you local machine.
   * Open a terminal with the path where you clone this repository.
   * Create a virtual environment, you can see the following [link](https://gist.github.com/mevangelista-alvarado/8ee2fd663e7446e543fc04eacce0f303) to know, how creates a virtual environment step by step.
   * Install the requirements, as follows:
      ```
      (venv_name) C:Users/dekstop/poisson$ pip install -r requirements.txt
      ```
   * We open the python terminal to start testing, as follows:
      ```
      (venv_name) C:Users/dekstop/poisson$ python
      ```

## Testing the class.
 * __Bivector to matrix function__
   We want convert to matrix the bivector $$\pi = x_{3}\partial x_{1}\wedge\partial x_2 - x_{2}\partial x_{1}\wedge\partial x_3 + x_{1}\partial x_{2}\wedge\partial x_2$$
   for this goal we use the `bivector_to_matrix` function
   ```
   >>> from poisson import PoissonGeometry
   >>> # We instantiate the Poisson class for dimension 3
   >>> poisson = ps(3)
   >>> poisson.bivector_to_matrix({12: 'x3', 13: '-x2', 23: 'x1'})
   Matrix([
   [  0,  x3, -x2],
   [-x3,   0,  x1],
   [ x2, -x1,   0]])
   ```
   Where the result is a Sympy Matrix type.
This function has an option for output is in latex syntax string, for this, we change flag `latex_syntax` to True, because its default value is False, as shown below.
   ```
   >>> print(poisson.bivector_to_matrix({12: 'x3', 13: '-x2', 23: 'x1'}, latex_syntax=True))
   \left[\begin{array}{ccc}0 & x_{3} & - x_{2}\\- x_{3} & 0 & x_{1}\\x_{2} & - x_{1} & 0\end{array}\right]
   ```

## Authors ✒️
This work is developed and maintained by:
 * **Pablo Suárez Serrato** - [@psuarezserrato](https://github.com/psuarezserrato)
 * **Jose Ruíz Pantaleón** - [@jcrpanta](https://github.com/jcrpanta)
 * **Miguel Evangelista Alvarado** - [@mevangelista-alvarado](https://github.com/mevangelista-alvarado)

## Licence 📄
__TODO__ Add licence

## Do not forget.
* Comment to others about this project 📢
* Cite this project if you use it 🤓 (__TODO__ add reference in Bibtex).
* Finally, if you know one of the authors, invite him a beer🍺.
---
