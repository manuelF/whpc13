/* La idea de este ejercicio es que practique 
el uso basico de la libreria thrust, y que 
compare las performances de la GPU y la CPU. 
*/

/* algunos headers de la libreria thrust */
// https://github.com/thrust/thrust/wiki/Documentation
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <string>
#include "timer.h"
using namespace std;


/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int N;
	if(argc==1) N=10000000;
	else N=atoi(argv[1]);	

	// HINT para los ejercicios: http://thrust.github.io/doc/structthrust_1_1multiplies.html

	// Declare/aloque tres vectores D1, D2 y D3 de tamanio N en la memoria de la GPU
	// SOLUCION:

	// llenar todo D1 con "0.0001 0.0001 ... 0.0001" 
	// SOLUCION:

	// llenar todo D2 con "0.1 1.1 2.1 ..." 
	// SOLUCION:

	/* ---- Start ---- */
	// un timer para GPU o CPU-multicore
	timer t;
	t.restart();
	
	int nveces=10;
	for(int i=0;i<nveces;i++)
	{
	// busque y aplique el algoritmo de thrust para multiplicar en la GPU: D3 = D1*D2
	// SOLUCION:

	}

	// imprimo tamano y tiempo de calculo en ms
	cout << "largo= " << N << ", tiempo [ms]= " << 1e3 * t.elapsed()/nveces << endl;
	return 0;
}
