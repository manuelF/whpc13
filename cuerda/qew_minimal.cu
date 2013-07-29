/*
=============================================================================================
A. B. Kolton

Segundo Encuentro Nacional de Computación de Alto Rendimiento para Aplicaciones Científicas
7 al 10 de Mayo de 2013.
=============================================================================================

Este programita simula la dinámica de una cuerdita elastica en un medio desordenado en la GPU, 
usando la librería Thrust, y la libreria Random123. Es una versión reducida de la que 
usamos con E. Ferrero y S. Bustingorry para la publicación:

Phys. Rev. E 87, 032122 (2013)
Nonsteady relaxation and critical exponents at the depinning transition
http://pre.aps.org/abstract/PRE/v87/i3/e032122
http://arxiv.org/abs/1211.7275

Y la explicación resumida del código esta en el material suplementario de la revista:
http://pre.aps.org/supplemental/PRE/v87/i3/e032122


Sin resolver los TODO ya se puede compilar con "make", y larga algunos timmings.
Genera dos ejecutables a la vez, uno de CUDA (corre en GPU) y otro de openMP (corre en CPU multicore).

OBJETIVOS:
- Practicar el manejo básico de la biblioteca Thrust.
- Practicar el manejo básico de la biblioteca Random123.
- Comparar performances CPU vs GPU.
- Aprender a combinar herramientas para resolver una ecuación diferencial 
parcial estocástica con desorden congelado.

EJERCICIOS:	
- Levantar los TODO.
- Para los mas expertos: Como mejoraría la performance del codigo?
*/


#include "timer.h"
#include <cmath>
#include <fstream>
#include <iostream>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG


/* algunos headers de la libreria thrust */
// https://github.com/thrust/thrust/wiki/Quick-Start-Guide
// https://github.com/thrust/thrust/wiki/Documentation
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

/* parámetros del problema */
#ifndef TAMANIO
#define L	4096   // numero de partículas/monomeros
#else
#define L	TAMANIO   // numero de partículas/monomeros
#endif
#ifndef TIEMPORUN
#define TRUN	100000       // numero de iteraciones temporales 
#else
#define TRUN	TIEMPORUN    // numero de iteraciones temporales 
#endif
#ifndef TPROP
#define TPROP	1000 // intervalo entre mediciones
#endif

#define F0	0.12       // fuerza uniforme sobre la interface/polímero 
#define Dt	0.1       // paso de tiempo
#define TEMP	0.003     // temperatura	
#define D0	1.0	  // intensidad del desorden
#define SEED 	12345678 // global seed RNG (quenched noise)
#define SEED2 	12312313 // global seed#2 RNG (thermal noise)

// para evitar poner "thrust::" a cada rato all llamar sus funciones
using namespace thrust;

// precisón elegida para los números reales
typedef double REAL;

// para generar números aleatorios gausianos a partir de dos uniformes
// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
__device__
REAL box_muller(RNG::ctr_type r_philox)
{
	// transforma el philox number a dos uniformes en (0,1]
 	REAL u1 = u01_open_closed_32_53(r_philox[0]);
  	REAL u2 = u01_open_closed_32_53(r_philox[1]);

  	REAL r = sqrt( -2.0*log(u1) );
  	REAL theta = 2.0*M_PI*u2;
	return r*sin(theta);    			
}

#ifdef PRINTCONFS
// para imprimir configuraciones cada tanto...
std::ofstream configs_out("configuraciones.dat");
void print_configuration(device_vector<REAL> &u, device_vector<REAL> &Ftot, float velocity, int size)
{
	long i=0;
	int every = int(L*1.0/size);
	for(i=1;i<u.size()-1;i+=every)
	{
		configs_out << u[i] << " " << Ftot[i] << " " << velocity << std::endl;
	}
	configs_out << "\n\n";
}
#endif

// functor usado para calcular la rugosidad 
struct roughtor: public thrust::unary_function<REAL,REAL>
{
    REAL u0; // un "estado interno" del functor	
    roughtor(REAL _u0):u0(_u0){};	
    __device__
    REAL operator()(REAL u)
    {	
	return (u-u0)*(u-u0);
    }
};	



/////////////////////////////////////////////////////////////////////////////
// FUNCTORS usados en los algoritmos TRANSFORM

struct fuerza
{
    RNG rng;       // random number generator
    long tiempo;   // parámetro
    REAL noiseamp; // parámetro
		
    fuerza(long _t):tiempo(_t)
    {
	noiseamp=sqrt(TEMP/Dt);
    }; 

    __device__
    REAL operator()(tuple<long,REAL,REAL,REAL> tt)
    {	
	// thread/particle id
	uint32_t tid=get<0>(tt); 

	// keys and counters 
    	RNG::ctr_type c={{}};
    	RNG::key_type k={{}};
	RNG::ctr_type r;

	k[0]=tid; 	  //  KEY = {threadID} 
	c[1]=SEED; 	  // COUNTER[1] = {a fijar mas tarde, GLOBAL SEED}

	// LAPLACIAN
	REAL um1=get<1>(tt);
	REAL u=get<2>(tt);
	REAL up1=get<3>(tt);
	REAL laplaciano = um1 + up1 - 2.*u;

	// DISORDER
#ifndef NODISORDER
	REAL firstRN, secondRN;
	int U=int(u);
	c[0]=uint32_t(U); // COUNTER={U,GLOBAL SEED}
	r = rng(c, k);
	firstRN=(u01_closed_closed_32_53(r[0])-0.5); // alternativa: box_muller(r);
	c[0]=uint32_t(U+1); // COUNTER={U+1,GLOBAL SEED}
	r = rng(c, k);
	secondRN=(u01_closed_closed_32_53(r[0])-0.5); // alternativa: box_muller(r);
	REAL quenched_noise = D0*(firstRN - (firstRN-secondRN)*(u-U));// linearly interpolated force
#endif
	// THERMAL NOISE
	REAL thermal_noise;

// TODO: agregar -DFINITETEMPERATURE en el Makefile para agregar 
// fluctuaciones termicas pero antes corregir el FIXME siguiente! 
#ifdef FINITETEMPERATURE
	// FIXME: Lo de abajo tiene un error grave!!. Corrijalo.
	c[0] = tid; // COUNTER = {tid, GLOBAL SEED #2} 
	c[1] = SEED2; // para evitar correlaciones entre el ruido térmico y el desorden congelado...
	r = rng(c, k);
	thermal_noise = noiseamp*box_muller(r);    			
#else
	thermal_noise=0.0;
#endif
	// Fuerza total en el monómero tid
	return (laplaciano+quenched_noise+thermal_noise+F0);
  }
};


// Explicit forward Euler step: lo mas simple que hay 
// (pero ojo con el paso de tiempo que no sea muy grande!)
struct euler
{
    __device__
    REAL operator()(REAL u_old, REAL force)
    {	
	return (u_old + force*Dt);
    }
};	


#ifdef OMP
#include <omp.h>
#endif

/////////////////////////////////////////////////////////////////////////////
int main(){

	#ifdef OMP
	std::cout << "#conociendo el host, OMP threads = " << omp_get_max_threads() << std::endl;
	#endif		

	/* containers e iteradores */

	// posiciones de los monómeros: 
	// Notar que alocamos dos elementos de mas para usarlos como "halo"
	// esto nos permite fijar las condiciones de borde facilmente, por ejemplo periódicas: u[0]=u[L]; u[L+1]=u[1];
	device_vector<REAL> u(L+2);

	// dos iteradores para definir el rango de interes para aplicar algoritmos 
	// (el +-1 nos permite descartar el halo en los algoritmos) 
	device_vector<REAL>::iterator u_it0 = u.begin()+1;
	device_vector<REAL>::iterator u_it1 = u.end()-1;

	// Si necesita el puntero "crudo" al array para pasarselo a un kernel de CUDA C/C++:
	REAL * u_raw_ptr = raw_pointer_cast(&u[1]);

	// container de fuerza total: 
	device_vector<REAL> Ftot(L); 
	// el rango de interés definido por los iteradores 
	device_vector<REAL>::iterator Ftot_it0 = Ftot.begin();
	device_vector<REAL>::iterator Ftot_it1 = Ftot.end();

	// alguna condición inicial chata (arbitraria)
	fill(u_it0,u_it1,0.0); 


	// simple (GPU/CPU) timer (curiosear el common/timer.h)
	timer t;
	double timer_fuerzas_elapsed=0.0;
	double timer_euler_elapsed=0.0;
	double timer_props_elapsed=0.0;

	// file para guardar algunas propiedades dependientes del tiempo
	#ifndef OMP
	std::ofstream propsout("someprops.dat");
	#else
	std::ofstream propsout("someprops_omp.dat");
	#endif

	// loop temporal
	//functor_fuerza fuerza(0);

	device_vector<REAL> u_old(L+2);
	for(long n=0;n<TRUN;n++)
	{
		// Impone PBC en el "halo"		
		u[0]=u[L];u[L+1]=u[1];


		t.restart(); // para cronometrar el tiempo de la siguiente transformación
		
		// Fuerza en cada monómero calculada concurrentemente en la GPU: 
		// Ftot(X)= laplacian + disorder + thermal_noise + F0, X=0,..,L-1
		// mirar el functor "fuerza" mas arriba... 
		// Notar: los iteradores de interés estan agrupado con un "fancy" zip_iterator, 
		// ya que transform no soporta mas de dos secuencias como input.
		// Notar: make_counting_iterator es otro "fancy" iterator, 
		// que simula una secuencia que en realidad no existe en memoria (implicit sequences).
		// https://github.com/thrust/thrust/wiki/Quick-Start-Guide
		// http://thrust.github.io/doc/group__fancyiterator.html
		//fuerza.set_time(n);
		transform(
			make_zip_iterator(make_tuple(
			make_counting_iterator<long>(0),u_it0-1,u_it0,u_it0+1
			)),
			make_zip_iterator(make_tuple(
			make_counting_iterator<long>(L),u_it1-1,u_it1,u_it1+1
			)),
			Ftot_it0,
			fuerza(n)
		);

		timer_fuerzas_elapsed+=t.elapsed();

		// Explicit forward Euler step, implementado en paralelo en la GPU: 
		// u(X,n) += Ftot(X,n) Dt, X=0,...,L-1
		// Mirar el functor "euler" mas arriba...
		// Notar: no hace falta zip_iterator, ya que transform si soporta hasta dos secuencias de input
        if(n%TPROP==0)
        {
            u_old=u;
        }

		t.restart();
		transform(
			u_it0,u_it1,Ftot_it0,u_it0, 
			euler()
		);
		timer_euler_elapsed+=t.elapsed();

		// algunas propiedades de interés, calculadas cada TPROP
		if(n%TPROP==0){		
			t.restart();
			
			/* TODO:
			   usando algoritmos REDUCE 
			   [ http://thrust.github.io/doc/group__reductions.html#gacf5a4b246454d2aa0d91cda1bb93d0c2 ]
			   calcule la velocidad media de la interface
			   y la posición del centro de masa de la interface 
			   HINT:
			   REAL velocity = reduce(....)/L; //center of mass velocity
			   REAL center_of_mass = reduce(....)/L; // center of mass position
			*/

			   REAL velocity = (reduce(           u.begin()+1,               u.end()-1,               0.0

               )
               +
               reduce(
               u_old.begin()+1,
               u_old.end()-1,
               0.0
               )
               )
               /L; //center of mass velocity
               
               /* Velocidad = delta Posicion / delta tiempo
                  SUM ( U(n)-U(n-1) ) /  Dt
                

               */
               

			   REAL center_of_mass = reduce(....)/L; // center of mass position
	               /*Centro de masas = Promedio de fuerzas
                  


               */
               

		/* TODO: 
			   usando el algoritmo TRANSFORM_REDUCE, 
			   [ http://thrust.github.io/doc/group__transformed__reductions.html#ga087a5af8cb83647590c75ee5e990ef66 ]
			   el functor "roughtor" arriba definido, 
			   y la posición del centro de masa "ucm" calculada en el TODO anterior, 
			   calcule la rugosidad (mean squared width) de la interface:
			   roughness := Sum_X [u(X)-ucm]^2 /L
			   HINT:
			   REAL roughness = transform_reduce(...,...,roughtor(center_of_mass),0.0,thrust::plus<REAL>());
			*/
			timer_props_elapsed+=t.elapsed();
	

			/* TODO: descomentar para que imprima la velocidad media, centro de masa, y rugosidad 
			   calculadas en los otros "TODOes", en el file "someprops.dat" */	
			//propsout << velocity << " " << center_of_mass << " " << roughness << std::endl;
			//propsout.flush();
			
			/* TODO:
			 Descomente -DPRINTCONFS en el Makefile, y recompile, para que imprima la posición 
			 y velocidad de 128 (o lo que quiera) particulas de la interface de tamanio L (una de cada L/128) 
			 en pantalla (descomente con cuidado, que imprime mucho!. Solo para hacer un "intuitive debugging")
			*/
			#ifdef PRINTCONFS
			print_configuration(u,Ftot,velocity,128);
			#endif
		}

		/* TODO: visualization!.
		   Si hay algun experimentado en el uso de, por ejemplo openGL, que se le ocurra como 
		   visualizar la línea en tiempo real en una máquina con el monitor conectado a la placa,
		   le agradeceré me enseñe :-). */
	}

	// resultados del timming
	double total_time = (timer_fuerzas_elapsed+timer_euler_elapsed+timer_props_elapsed); 

	std::cout << "L= " << L << " TRUN= " << TRUN << std::endl; 

	std::cout << "Forces calculation -> " << 1e3 * timer_fuerzas_elapsed << " miliseconds (" 
		  << int(timer_fuerzas_elapsed*100/total_time) << "%)" << std::endl;

	std::cout << "Euler step -> " << 1e3 * timer_euler_elapsed << " miliseconds (" 
		  << int(timer_euler_elapsed*100/total_time) << "%)" << std::endl;

	std::cout << "Properties -> " << 1e3 * timer_props_elapsed << " miliseconds (" 
		  << int(timer_props_elapsed*100/total_time) << "%)" << std::endl;

	std::cout << "Total -> " << 1e3 * total_time << " miliseconds (100%)" << std::endl;

	return 0;
}
