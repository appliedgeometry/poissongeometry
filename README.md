
# Poisson Geometry
Is a Python class to calculate classical methods in Poisson Geometry with symbolic calculus some of its function are:
 - [x] Poisson structures from Flaska-Ratiu method
 - [x] Schouten-Nijenhuis Bracket      
 - [x] Poisson Bracket
 - [x] The morphism sharp 
 - [x] Hamiltonian vector filed of a function respect to Poisson structure.

For more information you can read the [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki) this project. or the our [documentation]()

## Motivation 
This project results from the need to have somethings to take calculus in the Poisson Geometry, the following articles: 
 * [Poisson Structures on Smooth 4-Manifolds](https://www.researchgate.net/publication/263506998_Poisson_Structures_on_Smooth_4-Manifolds) by P. Suárez-Serrato, L.G. Naranjo & R. Vera, 
 * [Poisson structures on wrinkled fibrations](https://link.springer.com/article/10.1007/s40590-015-0072-8) by P. Suárez-Serrato & J. Torres Orozco, 
 * [Poisson and near-symplectic structures on generalized wrinkled fibrations in dimension 6](https://link.springer.com/article/10.1007/s10455-019-09651-2) by  P. Suárez-Serrato, J Torres Orozco & R. Vera
 * [On Bott-Morse Foliations and their Poisson Structures in Dimension 3](http://journalofsingularities.org/volume19/article2.html) by P. Suárez-Serrato, J Torres Orozco, R. Vera & M. Evangelista-Alvarado 

## Starting 🚀
#### You only have an interest in trying:
 * __On the cloud (without install nothing)__
   Please enter to us [codelab](https://colab.research.google.com/drive/1T2PG-vWaTrZ3Z5KK1U6-pK8uXQS7YdJu) 
   
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
	* Testing the class.
	   For example we want convert to matriz the bivector <a href="https://www.codecogs.com/eqnedit.php?latex=$$\pi=x_{3}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{2}}&space;-&space;x_{2}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{3}}&space;&plus;&space;x_{1}\frac{\partial}{\partial&space;x_{2}}\wedge\frac{\partial}{\partial&space;x_{3}}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\pi=x_{3}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{2}}&space;-&space;x_{2}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{3}}&space;&plus;&space;x_{1}\frac{\partial}{\partial&space;x_{2}}\wedge\frac{\partial}{\partial&space;x_{3}}$$" title="$$\pi=x_{3}\frac{\partial}{\partial x_{1}}\wedge\frac{\partial}{\partial x_{2}} - x_{2}\frac{\partial}{\partial x_{1}}\wedge\frac{\partial}{\partial x_{3}} + x_{1}\frac{\partial}{\partial x_{2}}\wedge\frac{\partial}{\partial x_{3}}$$" /></a>  
	   then <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> is equivalent to ```{12: 'x3', 13: '-x2', 23: 'x1'}``` in this class.
	   ```
	   >>> from poisson import PoissonGeometry
	   >>> # We instantiate the Poisson class for dimension 3
	   >>> pg = PoissonGeometry(3)
	   >>> pg.bivector_to_matrix({12: 'x3', 13: '-x2', 23: 'x1'})
	   Matrix([
	   [  0,  x3, -x2],
	   [-x3,   0,  x1],
	   [ x2, -x1,   0]])
	   ```
		
		This function has an option for output is in latex format string, for this, we change the flag latex_format to True (its default value is False) as shown below.
		
		```
		 >>> print(pg.bivector_to_matrix({12: 'x3', 13: '-x2', 23: 'x1'}, latex_format=True))
		   \left[\begin{array}{ccc}0 & x_{3} & - x_{2}\\- x_{3} & 0 & x_{1}\\x_{2} & - x_{1} & 	0\end{array}\right]
		```

		For more information to how use this class you can read the [documentation]() or the our [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki)

## TO DO 
Calculate Poisson Cohomology with linear coefficients.

## Bugs & Contributing 
Our issue tracker is at https://github.com/mevangelista-alvarado/poisson_geometry/issues. Please report any bugs that you find. Or, even better, If you interesting the project you can fork the repository on GitHub and create a pull request. We welcome all changes, big or small

## Licence 📄
MIT licence by authors + SymPy Development Team Licence +  BSD 3 license by Alan Bromborsky and GAlgebra Team 

## Authors ✒️
This work is developed and maintained by:
 * **Pablo Suárez Serrato** - [@psuarezserrato](https://github.com/psuarezserrato)
 * **Jose Ruíz Pantaleón** - [@jcrpanta](https://github.com/jcrpanta)
 * **Miguel Evangelista Alvarado** - [@mevangelista-alvarado](https://github.com/mevangelista-alvarado)

## Do not forget.
* Comment to others about this project 📢
* Cite this project if you use it 🤓.
* Finally, if you know one of the authors, invite him a beer🍺.
---
