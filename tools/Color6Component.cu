#include "Color6Component.cuh"


__device__ Color6Component::Color6Component()
{
	rComponent = 0;
	gComponent = 0;
	bComponent = 0;
}

__device__ Color6Component::Color6Component(int r, int g, int b)
{
	if (r < 0) rComponent = 0x7FFF;
	else if (r > 65535) rComponent = 0xFFFF;
	else rComponent = r;

	if (g < 0 || g > 65535) gComponent = 0x7FFF;
	else gComponent = g;

	if (b < 0 || b > 65535) bComponent = 0x7FFF;
	else bComponent = b;
}

__device__ Color6Component::~Color6Component()
{

}

__device__ Color6Component::Color6Component(int r, int g, int b, bool fromRGB) {
	if (fromRGB) {
		*this = Color6Component::Color6Component(r*128 + r/2, g*128 + g/2, b*128 + b/2);
	}
	else {
		*this = Color6Component::Color6Component(r, g, b);
	}
}

__device__ int Color6Component::getRGBComponent()
{
	unsigned int tempR = rComponent;
	unsigned int tempG = gComponent;
	unsigned int tempB = bComponent;

	if (tempR & 0x8000) {
		tempR = 0x7FFF;
	}
	tempR = tempR >> 7;
	if (tempG & 0x8000) {
		tempG = 0x7FFF;
	}
	tempG = tempG >> 7;
	if (tempB & 0x8000) {
		tempB = 0x7FFF;
	}
	tempB = tempB >> 7;

	tempG = tempG << 8;
	tempR = tempR << 16;
	tempR = tempR | tempG;
	tempR = tempR | tempB;
	return tempR;
}

__device__ Color6Component Color6Component::add(Color6Component arg)
{
	return Color6Component(rComponent + arg.rComponent, gComponent + arg.gComponent, bComponent + arg.bComponent);
}