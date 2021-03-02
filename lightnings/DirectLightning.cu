#include "DirectLightning.cuh"

__device__ DirectLightning::DirectLightning()
{

}

__device__ DirectLightning::DirectLightning(const DirectLightning& obj)
{
	*this = obj;
}