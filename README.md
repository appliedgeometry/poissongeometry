# Poisson Geometry
Es una clase de Python para hacer Cálculo Simbólico en Geometría de Poisson, algunas de sus funciones son:

 - [x] Obtener Estructuras de Poisson del tipo Flaska-Ratiu
 - [ ] Calcular Corchete de Schouten
 - [ ] Calcular Corchete de Poisson
 - [ ] Calcular Cohomología de Poisson

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
