#include "VoidDirectLightning.cuh"

__device__ VoidDirectLightning::VoidDirectLightning()
{
}
__device__ VoidDirectLightning::VoidDirectLightning(const DirectLightning& obj) : DirectLightning(obj)
{

}

__device__ VoidDirectLightning::~VoidDirectLightning()
{
}

__device__ VectorDouble VoidDirectLightning::getPos()
{
	return VectorDouble();
}

__device__ Color6Component VoidDirectLightning::getLightByDistance(double dist)
{
	return Color6Component();
}