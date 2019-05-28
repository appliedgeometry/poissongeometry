# Poisson Geometry
Es una clase de Python para hacer Cálculo Simbólico en Geometría de Poisson, algunas de sus funciones son:
 * Obtener Estructuras de Poisson del tipo Flaska-Ratiu
 * Calcular Corchete de Schouten
 * Calcular Corchete de Poisson
 * Calcular Cohomología de Poisson

## Pre-requisitos 📋
 * **[Python 3.7.x](https://www.python.org/)**
 * Dependencias
    * Sympy [Docs](https://docs.sympy.org/latest/index.html) & [Git-hub](https://github.com/sympy/sympy)
    * Galgebra [Docs](https://galgebra.readthedocs.io/en/latest/) & [Git-hub](https://github.com/pygae/galgebra)

## Instalación 🔧
Crear una carpeta en donde alojar en proyecto y después
Ejecuta en terminal
```
git clone https://github.com/mevangelista-alvarado/poisson_geometry.git
```
Despues crea un entorno virtual[¿como hacerlo?](https://gist.github.com/mevangelista-alvarado/8ee2fd663e7446e543fc04eacce0f303), para instalar las dependencias necesarias, ejecutando lo siguiente.
```
sudo python -m pip install -r requirements.txt
```

## Comenzando 🚀
#### Desde las consola.
Abre la terminal con direccion en la carpeta donde esta el archivo __poisson.py__ y ejecuta lo siguiente:
```
(venv)Users/desktop/poisson$ python
>>>
>>> from poisson import PoissonGeometry as ps
>>> import sympy
```
Probando las funciones
 * poisson_bracket
   ```
   # Instanciamos la clase Poisson
   >>> p = ps()
   # variables
   >>> x1,x2,x3 = sympy.symbols('x1 x2 x3')
   >>> M = sympy.Matrix([[0,x3,x2],[-x3,0,x1],[-x2,-x1,0]])
   >>> f = ['x1**2 + x3', 'x1 + x2 + x3']
   >>> poisson_bracket = p.poisson_bracket(M, f)
   # resultado
   >>> h
   2*x1*x3 - x1 + x2*(2*x1 - 1)
   ```# Poisson Geometry
Es una clase de Python para hacer Cálculo Simbólico en Geometría de Poisson, algunas de sus funciones son:
 [] Obtener Estructuras de Poisson del tipo Flaska-Ratiu
 [] Calcular Corchete de Schouten
 [x] Calcular Corchete de Poisson
 [] Calcular Cohomología de Poisson

## Pre-requisitos 📋
 * **[Python 3.7.x](https://www.python.org/)**
 * Dependencias
    * Sympy [Docs](https://docs.sympy.org/latest/index.html) & [Git-hub](https://github.com/sympy/sympy)
    * Galgebra [Docs](https://galgebra.readthedocs.io/en/latest/) & [Git-hub](https://github.com/pygae/galgebra)

## Instalación 🔧
Abre la terminal en la dirección donde vayas a guardar el proyecto (por ejemplo tu escritorio) y ejecuta lo siguiente:   
```
C:Users/dekstop/poisson$ git clone https://github.com/mevangelista-alvarado/poisson_geometry.git
```
Después crea un entorno virtual (¿Cómo hacerlo?)[Da click acá](https://gist.github.com/mevangelista-alvarado/8ee2fd663e7446e543fc04eacce0f303), para instalar las dependencias necesarias, ejecutando lo siguiente. 
```
C:Users/dekstop/poisson$ sudo python -m pip install -r requirements.txt
```

## Comenzando 🚀
#### Desde las consola.
Abre la terminal con direccion en la carpeta donde esta el archivo __poisson.py__ y ejecuta lo siguiente:
```
C:Users/dekstop/poisson$ python 
>>> from poisson import PoissonGeometry as ps 
>>> import sympy
# Instanciamos la clase Poisson
>>> p = ps()
# variables 
>>> x1,x2,x3 = sympy.symbols('x1 x2 x3')
>>> M = sympy.Matrix([[0,x3,x2],[-x3,0,x1],[-x2,-x1,0]])
```
Probando las funciones
 * poisson_bracket
   ```
   >>> f = ['x1**2 + x3', 'x1 + x2 + x3'] # funciones para aplicar el corchete
   >>> poisson_bracket = p.poisson_bracket(M, f)
   >>> poisson_bracket # resultado
   >>> 2*x1*x3 - x1 + x2*(2*x1 - 1)
   ```
* hamiltonian_vector_field
   ```
   >>> f = 'x1**2 + x3'
   >>> hamiltonian_vector_field = p.hamiltonian_vector_field(M, f)
   >>> hamiltonian_vector_field # Resultado 
   >>> -x2*Dx1 + x1*(2*x3 - 1)*Dx2 + 2*x1*x2*Dx3
   ```
* pi_sharp_morphism
   ```
   >>> w = ['x1','x2','x3**2']
   >>> pi_sharp_morphism = p.pi_sharp_morphism(M, w)
   >>> pi_sharp_morphism # resultado
   >>> -x2*x3*(x3 + 1)*dx1 + x1*x3*(1 - x3)*dx2 + 2*x1*x2*dx3
   ```

## Autores ✒️
Este trabajo es desarrollado y mantenido por:
 * **Pablo Suárez Serrato** - [@psuarezserrato](https://github.com/psuarezserrato)
 * **Jose Ruíz** - [@jcrpanta](https://github.com/jcrpanta)
 * **Miguel Evangelista** - [@mevangelista-alvarado](https://github.com/mevangelista-alvarado)

## Licencia 📄
Próximamente

## No Olvides.
* Comentar a otros sobre este proyecto 📢
* Citar este proyecto si lo utilizas 🤓 (Próximamente referencia en formato Bibtex).
* Por último, si conoces a uno de los autores invitale una cerveza 🍺.
---


## Autores ✒️
Este trabajo es desarrollado y mantenido por:
 * **Pablo Suárez Serrato** - [@psuarezserrato](https://github.com/psuarezserrato)
 * **Jose Ruíz** - [@jcrpanta](https://github.com/jcrpanta)
 * **Miguel Evangelista** - [@mevangelista-alvarado](https://github.com/mevangelista-alvarado)

## Licencia 📄
Próximamente

## No Olvides.
* Comentar a otros sobre este proyecto 📢
* Citar este proyecto si lo utilizas 🤓 (Próximamente referencia en formato Bibtex).
* Por último, si conoces a uno de los autores invitale una cerveza 🍺.
---
