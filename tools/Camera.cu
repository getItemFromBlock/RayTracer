#include "Camera.cuh"

__device__ Camera::Camera(VectorDouble lookfrom, VectorDouble direction, VectorDouble vup, double vfov, double aspect_ratio)
{
	double theta = toRadians(vfov);
	double h = tan(theta/2.0);
	double viewport_height = 2.0*h;
	double viewport_width = aspect_ratio * viewport_height;

	VectorDouble w = direction.unitVector();
	VectorDouble u = vup.cross(w).unitVector();
	VectorDouble v = w.cross(u);

	origin = VectorDouble();
	origin = VectorDouble(lookfrom.getX(), lookfrom.getY(), lookfrom.getZ());
	horizontal = VectorDouble();
	horizontal = u.mul(viewport_width);
	vertical = VectorDouble();
	vertical = v.mul(viewport_height);
	lower_left_corner = VectorDouble();
	lower_left_corner = origin.sub(horizontal.div(2.0)).sub(vertical.div(2.0)).sub(w);

}

__device__ Camera::Camera()
{
	origin = VectorDouble();
	horizontal =  VectorDouble();
	vertical = VectorDouble();
	lower_left_corner = VectorDouble();
}

__device__ Camera::~Camera()
{
}

__device__ void Camera::refresh(VectorDouble lookfrom, VectorDouble direction, VectorDouble vup, double vfov, double aspect_ratio)
{
	double theta = toRadians(vfov);
	double h = tan(theta / 2.0);
	double viewport_height = 2.0*h;
	double viewport_width = aspect_ratio * viewport_height;

	VectorDouble w = direction.unitVector();
	VectorDouble u = vup.cross(w).unitVector();
	VectorDouble v = w.cross(u);

	origin = VectorDouble(lookfrom.getX(), lookfrom.getY(), lookfrom.getZ());
	horizontal = u.mul(viewport_width);
	vertical = v.mul(viewport_height);
	lower_left_corner = origin.sub(horizontal.div(2.0)).sub(vertical.div(2.0)).sub(w);
}

__device__ Ray Camera::getRay(double u, double v)
{
	return Ray(VectorDouble(origin.getX(), origin.getY(), origin.getZ()), lower_left_corner.add(horizontal.mul(u)).add(vertical.mul(v)).sub(origin));
}

__device__ double Camera::toRadians(double input) {
	return input / 180.0*CUDART_PI;
}