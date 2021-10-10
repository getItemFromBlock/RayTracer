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

__device__ Vector3D VoidDirectLightning::getPos()
{
	return Vector3D();
}

__device__ Color6Component VoidDirectLightning::getLightByDistance(float dist)
{
	return Color6Component();
}