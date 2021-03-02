#include "VectorDouble.cuh"

__host__ __device__ VectorDouble::VectorDouble(double arg1, double arg2, double arg3) // constructeur - définition
{
	X = arg1;
	Y = arg2;
	Z = arg3;
}

__host__ __device__ VectorDouble::VectorDouble()
{
	X = 0;
	Y = 0;
	Z = 0;
}

__host__ __device__ VectorDouble::~VectorDouble() // Destructeur - définition
{

}

__host__ __device__ VectorDouble VectorDouble::operator=(const VectorDouble other)
{
	if (this != &other) {
		X = other.X;
		Y = other.Y;
		Z = other.Z;
	}
	return *this;
}

__host__ __device__ void VectorDouble::setX(double arg)
{
	X = arg;
}

__host__ __device__ void VectorDouble::setY(double arg)
{
	Y = arg;
}

__host__ __device__ void VectorDouble::setZ(double arg)
{
	Z = arg;
}

__host__ __device__ double VectorDouble::getX()
{
	return X;
}

__host__ __device__ double VectorDouble::getY()
{
	return Y;
}

__host__ __device__ double VectorDouble::getZ()
{
	return Z;
}

__host__ __device__ double VectorDouble::getLength()
{
	return sqrt((X*X) + (Y*Y) + (Z*Z));
}

__host__ __device__ VectorDouble VectorDouble::add(VectorDouble v)
{
	return VectorDouble(X + v.X, Y + v.Y, Z + v.Z);
}

__host__ __device__ VectorDouble VectorDouble::sub(VectorDouble v)
{
	return VectorDouble(X - v.X, Y - v.Y, Z - v.Z);
}

__host__ __device__ VectorDouble VectorDouble::mul(VectorDouble v)
{
	return VectorDouble(X * v.X, Y * v.Y, Z * v.Z);
}

__host__ __device__ VectorDouble VectorDouble::mul(double d)
{
	return VectorDouble(X * d, Y * d, Z * d);
}

__host__ __device__ VectorDouble VectorDouble::div(double d)
{
	if (d == 0)
		return mul(1e50);
	return mul(1.0 / d);
}

__host__ __device__ double VectorDouble::dot(VectorDouble v)
{
	return X * v.X + Y * v.Y + Z * v.Z;
}

__host__ __device__ VectorDouble VectorDouble::cross(VectorDouble v)
{
	return VectorDouble((Y*v.Z) - (Z*v.Y), (Z*v.X) - (X*v.Z), (X*v.Y) - (Y*v.X));
}

__host__ __device__ VectorDouble VectorDouble::unitVector()
{
	return div(getLength());
}

__host__ __device__ double VectorDouble::length_squared()
{
	return (X * X) + (Y * Y) + (Z * Z);
}