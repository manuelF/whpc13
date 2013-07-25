/*
=============================================================================================
A. B. Kolton

Segundo Encuentro Nacional de Computación de Alto Rendimiento para Aplicaciones Científicas
7 al 10 de Mayo de 2013.
=============================================================================================

Este programita transforma Fourier una señal unidimensional, almacenada en GPU, usando cufft.
Se puede compilar y correr directamente, midiendo el tiempo de GPU, e imprimiendo la transformada.

$ nvcc simple_cufft_thrust.cu -lcufft -arch=sm_20 -o simple_cufft

OBJETIVOS:
- Practicar el manejo básico de la librería cuFFT.
- Practicar el manejo básico de la librería Thrust.
- Practicar la interoperabilidad cuFFT-Thrust.

EJERCICIOS:
- Utilizar simple y doble precisión, y diferentes tamaños. Comparar performances.
- Levantar los TODO.
- Importante: ¿Cómo están ordenadas las frecuencias de la transformada? 
*/


/* algunos headers de la libreria thrust */
// https://github.com/thrust/thrust/wiki/Documentation
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <fstream>
#include "cutil.h"	// CUDA_SAFE_CALL, CUT_CHECK_ERROR
#include "timer.h"


// CUFFT include http://docs.nvidia.com/cuda/cufft/index.html
#include <cufft.h>

using namespace std;

/* + Array Size N: use only powers of 2 */
#ifndef TAMANIO
#define N 1048576
#else
#define N TAMANIO
#endif

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef cufftDoubleReal REAL;
typedef cufftDoubleComplex COMPLEX;
#else
typedef cufftReal REAL;
typedef cufftComplex COMPLEX;
#endif

/* Parametros de la senial */
#define A1 4
#define A2 6
#define T1 N/4
#define T2 N/8
struct FillSignal
{
	__device__ __host__ 
	REAL operator()(unsigned tid)
    	{	
		// ponga aqui su funcion preferida...
		return A1*2.0*cosf(2*M_PI*tid*T1/(float)N) + A2*2.0*sinf(2*M_PI*tid*T2/(float)N);
    	}
};

#ifdef OMP
#include <omp.h>
#endif

///////////////////////////////////////////////////////////////////////////
int main(void) {

	#ifdef OMP
	std::cout << "#conociendo el host, OMP threads = " << omp_get_max_threads() << std::endl;
	#endif

	// Un container de thrust para guardar el input real en GPU 
	thrust::device_vector<REAL> D_input(N);

	// toma el raw_pointer del array de input, para pasarselo a CUFFT luego
	REAL *d_input = thrust::raw_pointer_cast(&D_input[0]);

	// Un container de thrust para guardar el ouput complejo en GPU = transformada del input 
	int Ncomp=N/2+1;
	thrust::device_vector<COMPLEX> D_output(Ncomp);

	// toma el raw_pointer del array de output, para pasarselo a CUFFT luego
	COMPLEX *d_output = thrust::raw_pointer_cast(&D_output[0]); 

	// crea el plan de transformada de cuFFT
	#ifdef DOUBLE_PRECISION
	cufftHandle plan_d2z;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_d2z,N,CUFFT_D2Z,1));
	#else
	cufftHandle plan_r2c;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_r2c,N,CUFFT_R2C,1));
	#endif

	//lleno array de tamanio N con la senial, a travez del functor "FillSignal"
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),D_input.begin(),FillSignal());

	/* ---- Start ---- */
	// un timer para GPU
	timer t;
	t.restart();

	//Transforma Fourier ejecutando el plan
	#ifdef DOUBLE_PRECISION
	CUFFT_SAFE_CALL(cufftExecD2Z(plan_d2z, d_input, d_output));
	#else
	CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, d_input, d_output));
	#endif

	double t_elapsed=t.elapsed();
	/* ---- Stop ---- */

	// declara un vector para copiar/guardar la transformada en el host:
	thrust::host_vector<COMPLEX> H_output=D_output;

	/* Imprime la transformada */
	cout << "# Tamanio del array = " << N << endl;
	cout << "# Tiempo de GPU " << 1e3*t_elapsed << " miliseconds" << endl;
#ifdef IMPRIMIR
	ofstream transformada_out("transformada.dat");
	for(int j = 0 ; j < Ncomp ; j++){
		transformada_out << COMPLEX(H_output[j]).x << " " << COMPLEX(H_output[j]).y << endl;
	}
    transformada_out.close();
#endif

// TODO: Verifique que el resultado sea correcto, por ejemeplo, usando seniales cuya transformada conoce analiticamente 
// o verificando simetrias, etc. Preste atencion al ordenamiento de las frecuencias...
// HINT: sin(x)=[e^{ix}-e^{-ix}]/2i , cos(x)=[e^{ix}+e^{-ix}]/2 y mire la formula de la antitransformada (clase)

// TODO: 
// Agregue planes para realizar la antitransformada de la senial con CUFFT (contemple los casos double y float)
//	#ifdef DOUBLE_PRECISION
//	#else
//	#endif
	#ifdef DOUBLE_PRECISION
	cufftHandle plan_z2d;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_z2d,N,CUFFT_Z2D,1));
	#else
	cufftHandle plan_c2r;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_c2r,N,CUFFT_C2R,1));
	#endif

// TODO: 
// Declare/aloque un container de Thrust para guardar la antitransformada, y el raw_pointer para pasar a CUFFT
	thrust::device_vector<REAL> D_AntiTransformed(N);

	// toma el raw_pointer del array de output, para pasarselo a CUFFT luego
	REAL *d_Anti = thrust::raw_pointer_cast(&D_AntiTransformed[0]); 


// TODO:
// Ejecute los planes cuFFT de la antitransformada (contemple los casos double y float)
//	#ifdef DOUBLE_PRECISION
//	#else
//	#endif

	#ifdef DOUBLE_PRECISION
	CUFFT_SAFE_CALL(cufftExecZ2D(plan_z2d, d_output, d_Anti));
	#else
	CUFFT_SAFE_CALL(cufftExecC2R(plan_c2r, d_output, d_Anti));
	#endif


// TODO: 
// Declare/aloque dos containers de Thrust: uno para guardar la antitransformada, y otro para el input original, en el host
// y copie los respectivos contenidos del device al host

	thrust::host_vector<REAL> Original_input=D_input;
	thrust::host_vector<REAL> AntiTransformed_output=D_AntiTransformed;


// TODO:
// Imprima en un file el input original y la antitransformada de la transformada, para comparar
#ifdef IMPRIMIR
	ofstream comparativa_out("comparativa.dat");
	for(int j = 0 ; j < N ; j++){
		comparativa_out << Original_input[j] << "\t " << AntiTransformed_output[j] << endl;
	}
    comparativa_out.close();
#endif
	#ifdef DOUBLE_PRECISION
    cufftDestroy(plan_d2z);
    cufftDestroy(plan_z2d);

    #else
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    #endif
	return 0;
}
