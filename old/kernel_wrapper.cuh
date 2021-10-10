#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

// Wrapper permettant l'exécution du kernel depuis une classe C++
namespace Wrapper {
	void wrapper(void);
}