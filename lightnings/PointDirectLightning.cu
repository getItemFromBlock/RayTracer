#include "PointDirectLightning.cuh"

__device__ PointDirectLightning::PointDirectLightning()
{
	center = VectorDouble();
	light = Color6Component();
	atenuation = 100.0;
}

__device__ PointDirectLightning::PointDirectLightning(VectorDouble origin, Color6Component lightcolor, double atenuationByDist)
{
	center = origin;
	light = lightcolor;
	atenuation = atenuationByDist;
}

__device__ PointDirectLightning::PointDirectLightning(const DirectLightning& obj) : DirectLightning(obj)
{

}

__device__ PointDirectLightning::~PointDirectLightning()
{
}

__device__ VectorDouble PointDirectLightning::getPos()
{
	return center;
}

__device__ Color6Component PointDirectLightning::getLightByDistance(double dist)
{
	double r = light.rComponent*powf(atenuation / 100.0, dist);
	double g = light.gComponent*powf(atenuation / 100.0, dist);
	double b = light.bComponent*powf(atenuation / 100.0, dist);
	return Color6Component(r, g, b);
}