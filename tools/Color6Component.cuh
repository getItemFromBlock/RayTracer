#pragma once
#include <cuda_runtime.h>

class Color6Component
{
public:
	__device__ Color6Component();
	__device__ Color6Component(int r, int g, int b);
	__device__ ~Color6Component();
	__device__ Color6Component(int r, int g, int b, bool fromRGB);

	unsigned short int rComponent;
	unsigned short int gComponent;
	unsigned short int bComponent;

	__device__ int getRGBComponent();
	__device__ Color6Component add(Color6Component arg);
};