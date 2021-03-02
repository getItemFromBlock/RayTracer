#include "AmbientLightning.cuh"


__device__ AmbientLightning::AmbientLightning()
{
	light = Color6Component();
}

__device__ AmbientLightning::AmbientLightning(Color6Component arg)
{
	light = arg;
}

__device__ AmbientLightning::~AmbientLightning()
{

}

__device__ void AmbientLightning::setLight(Color6Component arg)
{
	light = arg;
}

__device__ Color6Component AmbientLightning::getLight()
{
	return light;
}