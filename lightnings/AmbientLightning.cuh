#pragma once
#include "../tools/Vector3D.cuh"
#include "../tools/Color6Component.cuh"
class AmbientLightning
{
public:
	__device__ AmbientLightning();
	__device__ AmbientLightning(Color6Component arg);	// Constructeur - d�claration
	__device__ ~AmbientLightning();		// Destructeur - d�claration
	
	__device__ Color6Component getLight();
	__device__ void setLight(Color6Component arg);
private:
	Color6Component light;
};