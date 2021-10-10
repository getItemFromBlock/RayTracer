#include "Camera.cuh"

__device__ Camera::Camera(Vector3D lookfrom, Vector3D direction, Vector3D vup, float vfov, float aspect_ratio)
{
	float theta = toRadians(vfov);
	float h = tan(theta/2.0);
	float viewport_height = 2.0*h;
	float viewport_width = aspect_ratio * viewport_height;

	Vector3D w = direction.unitVector();
	Vector3D u = vup.cross(w).unitVector();
	Vector3D v = w.cross(u);

	origin = Vector3D(lookfrom.getX(), lookfrom.getY(), lookfrom.getZ());
	horizontal = u.mul(viewport_width);
	vertical = v.mul(viewport_height);
	lower_left_corner = origin.sub(horizontal.div(2.0)).sub(vertical.div(2.0)).sub(w);

}

__device__ Camera::Camera()
{
	origin = Vector3D();
	horizontal = Vector3D();
	vertical = Vector3D();
	lower_left_corner = Vector3D();
}

__device__ Camera::~Camera()
{
}

__device__ void Camera::refresh(Vector3D lookfrom, Vector3D direction, Vector3D vup, float vfov, float aspect_ratio)
{
	float theta = toRadians(vfov);
	float h = tan(theta / 2.0);
	float viewport_height = 2.0*h;
	float viewport_width = aspect_ratio * viewport_height;

	Vector3D w = direction.unitVector();
	Vector3D u = vup.cross(w).unitVector();
	Vector3D v = w.cross(u);

	origin = Vector3D(lookfrom.getX(), lookfrom.getY(), lookfrom.getZ());
	horizontal = u.mul(viewport_width);
	vertical = v.mul(viewport_height);
	lower_left_corner = origin.sub(horizontal.div(2.0)).sub(vertical.div(2.0)).sub(w);
}

__device__ Ray Camera::getRay(float u, float v)
{
	return Ray(Vector3D(origin.getX(), origin.getY(), origin.getZ()), lower_left_corner.add(horizontal.mul(u)).add(vertical.mul(v)).sub(origin));
}

__device__ float Camera::toRadians(float input) {
	return input / 180.0*CUDART_PI;
}