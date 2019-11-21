
# `PoissonGeometry`
A Python class for (local) Poisson-Nijenhuis calculus on Poisson manifolds:

|  sharp_morphism            | lichnerowicz_poiison_operador  | modular_vf           |
| :-------:                  | :------:                       | :-----:              |
| poisson_bracket            | bivector_to_matrix             | jacobiator           |
| __hamiltonian_vf__         | __is_homogeneaos_unimodular__  | __one_forms_bracket__|
| isomorphic_lie_poisson_R3  | linear_normal_form_R3          | gauge_transformation |
|__flaschka_ratiu_bivector__ | __is_poisson_tensor__          |__is_in_kernel__      |
|is_casimir                  |is_poisson_vf                   |is_poisson_par        |
|__curl_operator__           |                                |                      |

<!-- For more information you can read the [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki) this project. or the our [documentation]()-->

## Motivation 
As of today, different programs that facilitate tensor calculus has been developed. Indeed, Python and some math softwares such as Maple, Mathematica, Maxima or Sage have implemented specialized modules in this kind of calculus. However, the literature and publicly available software packages for symbolic computations with Poisson structures is quite scarce.

 * L.  C.  Garcia-Naranjo,  P.  Suárez-Serrato  and  R.  Vera, [Poisson Structures on Smooth 4-Manifolds](https://link.springer.com/article/10.1007/s11005-015-0792-8), Lett. Math. Phys. 105, 1533-1550 (2015). 
 * P. Suárez-Serrato and J. Torres-Orozco. [Poisson structures on wrinkled fibrations](https://link.springer.com/article/10.1007/s40590-015-0072-8) Bol. Soc.Mat. Mex. 22, 263-280 (2016) 
 * Suárez-Serrato, P., Torres Orozco, J. & Vera, R.  [Poisson and near-symplectic structures on generalized wrinkled fibrations in dimension 6](https://link.springer.com/article/10.1007/s10455-019-09651-2) Ann Glob Anal Geom (2019) 55, 777-804 (2019)
 * M. Evangelista-Alvarado, P. Suárez-Serrato, J. Torres-Orozco and R. Vera. [On Bott-Morse Foliations and their Poisson Structures in Dimension 3](http://journalofsingularities.org/volume19/article2.html), Journal of Singularities 19, 19-33(2019)

## Starting 🚀
#### Trying:
 * __On Colab__ [Spanish](https://colab.research.google.com/drive/1SN6PS0auO-h3aCXIenblnwJIV-YpowtZ)/ [English](https://colab.research.google.com/drive/1XYcaJQ29XwkblXQOYumT1s8_00bHUEKZ)
   
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
	   then <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> is equivalent to ```{(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}``` in this class.
	   ```
	   >>> from poisson import PoissonGeometry
	   >>> # We instantiate the Poisson class for dimension 3
	   >>> pg = PoissonGeometry(3)
	   >>> pg.bivector_to_matrix({(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'})
	   Matrix([
	   [  0,  x3, -x2],
	   [-x3,   0,  x1],
	   [ x2, -x1,   0]])
	   ```
		
		This function has an option for output is in latex format string, for this, we change the flag latex_format to True (its default value is False) as shown below.
		
		```
		 >>> print(pg.bivector_to_matrix({(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}, latex_format=True))
		   \left[\begin{array}{ccc}0 & x_{3} & - x_{2}\\- x_{3} & 0 & x_{1}\\x_{2} & - x_{1} & 0\end{array}\right]
		```
		<!--For more information to how use this class you can read the [documentation]() or the our [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki)-->

<!--## TO DO
Calculate Poisson Cohomology with linear coefficients.-->

## Bugs & Contributing
Our issue tracker is at https://github.com/mevangelista-alvarado/poisson_geometry/issues. Please report any bugs that you find. Or, even better, If you interesting the project you can fork the repository on GitHub and create a pull request. We welcome all changes, big or small

## Licence 📄
MIT licence

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
