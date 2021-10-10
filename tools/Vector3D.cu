#include "Vector3D.cuh"

__host__ __device__ Vector3D::Vector3D(float arg1, float arg2, float arg3) // constructeur - définition
{
	X = arg1;
	Y = arg2;
	Z = arg3;
}

__host__ __device__ Vector3D::Vector3D()
{
	X = 0;
	Y = 0;
	Z = 0;
}

__host__ __device__ Vector3D::~Vector3D() // Destructeur - définition
{

}

__host__ __device__ Vector3D Vector3D::operator=(const Vector3D other)
{
	if (this != &other) {
		X = other.X;
		Y = other.Y;
		Z = other.Z;
	}
	return *this;
}

__host__ __device__ void Vector3D::setX(float arg)
{
	X = arg;
}

__host__ __device__ void Vector3D::setY(float arg)
{
	Y = arg;
}

__host__ __device__ void Vector3D::setZ(float arg)
{
	Z = arg;
}

__host__ __device__ float Vector3D::getX()
{
	return X;
}

__host__ __device__ float Vector3D::getY()
{
	return Y;
}

__host__ __device__ float Vector3D::getZ()
{
	return Z;
}

__host__ __device__ float Vector3D::getLength()
{
	return sqrtf((X*X) + (Y*Y) + (Z*Z));
}

__host__ __device__ Vector3D Vector3D::add(Vector3D v)
{
	return Vector3D(X + v.X, Y + v.Y, Z + v.Z);
}

__host__ __device__ Vector3D Vector3D::sub(Vector3D v)
{
	return Vector3D(X - v.X, Y - v.Y, Z - v.Z);
}

__host__ __device__ Vector3D Vector3D::mul(Vector3D v)
{
	return Vector3D(X * v.X, Y * v.Y, Z * v.Z);
}

__host__ __device__ Vector3D Vector3D::mul(float d)
{
	return Vector3D(X * d, Y * d, Z * d);
}

__host__ __device__ Vector3D Vector3D::div(float d)
{
	if (d == 0)
		return mul((float)(1e20));
	return mul(1.0 / d);
}

__host__ __device__ float Vector3D::dot(Vector3D v)
{
	return X * v.X + Y * v.Y + Z * v.Z;
}

__host__ __device__ Vector3D Vector3D::cross(Vector3D v)
{
	return Vector3D((Y*v.Z) - (Z*v.Y), (Z*v.X) - (X*v.Z), (X*v.Y) - (Y*v.X));
}

__host__ __device__ Vector3D Vector3D::unitVector()
{
	return div(getLength());
}

__host__ __device__ float Vector3D::length_squared()
{
	return (X * X) + (Y * Y) + (Z * Z);
}