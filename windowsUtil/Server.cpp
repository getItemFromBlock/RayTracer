#include <Windows.h>
#include "WindowsHolder.h"
#include "../tools/ThreadArgs.h"
#include "../tools/Ray.cuh"
#include <chrono>
#include <thread>
#include <cmath>
#include <corecrt_math_defines.h>


static __int64 timeSinceEpochMillisecS() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

constexpr float speedBase = 1.0 / 20.0; //Speeds are in m/s, then converted for 50 tps
constexpr float speedCrouch = 0.55 / 20.0;
constexpr float gravAcc = 0.1 / 20.0;
constexpr float jumpPower = 0.12;
static WindowsHolder* windowS;
static ThreadArgs* sArgs;
static ObjectsHolder* serverHolder;
static Vector3D playerSpeed = Vector3D();

static float toRadiansS(float angle) {
	return angle / 180 * M_PI;
}

static Vector3D getPlayerDirS() {
	Vector3D result = Vector3D(0, 0, 0);
	const float speed = (windowS->getInputValue(VK_CONTROL) ? speedCrouch : speedBase);
	if (windowS->getInputValue('Z')) {
		result.setX(1.0);
	}
	if (windowS->getInputValue('S')) {
		result.setX(result.getX() - 1.0);
	}
	if (windowS->getInputValue('D')) {
		result.setZ(1.0);
	}
	if (windowS->getInputValue('Q')) {
		result.setZ(result.getZ() - 1.0);
	}
	if (result.length_squared() > 0.001) {
		result = result.unitVector().mul(speed);
	}
	return result;
}

static bool collidePlayerWSphere(int* args, int index, Vector3D pos, Vector3D* dest) {
	if (args[index + 13] == 1) return false;
	float posX = *(float*)(&args[index + 1]);
	float posY = *(float*)(&args[index + 2]);
	float posZ = *(float*)(&args[index + 3]);
	float rad = *(float*)(&args[index + 4]);

	if (dest->getY() < posY + rad && dest->getY() > posY) {
		Vector3D pTemp1 = Vector3D(posX,0,posZ);
		Vector3D pTemp2 = Vector3D(dest->getX(),0,dest->getZ());
		float pTemp3 = pTemp1.sub(pTemp2).getLength();
		float pTemp4 = posY + rad - (0.5 * pTemp3);
		if (pTemp3 < rad + 0.2 && dest->getY() < pTemp4) {
			dest->setY(pTemp4);
			return true;
		}
	}
	return false;
}

static bool collidePlayerWTriangle(int* args, int index, Vector3D pos, Vector3D* dest) {
	Vector3D A = Vector3D(*(float*)(&args[index + 1]), *(float*)(&args[index + 2]), *(float*)(&args[index + 3]));
	Vector3D B = Vector3D(*(float*)(&args[index + 4]), *(float*)(&args[index + 5]), *(float*)(&args[index + 6]));
	Vector3D C = Vector3D(*(float*)(&args[index + 7]), *(float*)(&args[index + 8]), *(float*)(&args[index + 9]));

	Vector3D AB = B.sub(A);
	Vector3D AC = C.sub(A);
	Vector3D normal = AB.cross(AC);
	Vector3D pos2 = Vector3D(pos.getX(), pos.getY() + 0.5, pos.getZ());
	Ray r = Ray(pos2,Vector3D(0,-1,0));
	float det = -r.getDirection().dot(normal);
	float invdet = 1.0 / det;
	Vector3D AO = r.getOrigin().sub(A);
	Vector3D DAO = AO.cross(r.getDirection());
	float u = AC.dot(DAO) * invdet;
	float v = -AB.dot(DAO) * invdet;
	float t = AO.dot(normal) * invdet;
	if (det >= 1e-6 && t >= 0 && t <= 0.6 && u >= -0.2 && v >= -0.2 && u + v <= 1.2) {
		float ty = r.getOrigin().add(r.getDirection().mul(t)).getY();
		float ty2 = dest->getY();
		dest->setY(ty>ty2?ty:ty2);
		return true;
	}
	return false;
}


static bool collidePlayer(Vector3D pos, Vector3D* dest) {
	int dS = serverHolder->getDrawablesSize();
	if (dS > 0) {
		int* obj = serverHolder->getDrawables();
		int objS = serverHolder->objectSizeA;
		bool collide = false;
		for (int i = 0; i < dS; i++) {
			int index = objS * i;
			if (obj[index] == 1 || obj[index] == 4) {
				if (collidePlayerWSphere(obj, index, pos, dest)) {
					collide = true;
				}
			}
			if (obj[index] == 2 || obj[index] == 3) {
				if (collidePlayerWTriangle(obj, index, pos, dest)) {
					collide = true;
				}
			}
		}
		return collide;
	}
	return false;
}

static DWORD WINAPI serverMain(LPVOID lpParameter) {
	sArgs = ((ThreadArgs*)lpParameter);
	windowS = sArgs->getWHolder();
	serverHolder = sArgs->getOHolder();

	int tickTime = 0;
	long long int dt = timeSinceEpochMillisecS();
	long long int dt2 = 0;

	while (true) {
		Vector3D playerDir = getPlayerDirS();
		Vector3D dest = Vector3D();
		playerSpeed.setX(cos(toRadiansS(windowS->rotX)) * (-playerDir.getX()) + sin(toRadiansS(windowS->rotX)) * playerDir.getZ());
		playerSpeed.setZ(sin(toRadiansS(windowS->rotX)) * (-playerDir.getX()) - cos(toRadiansS(windowS->rotX)) * playerDir.getZ());
		float vy = playerSpeed.getY();
		playerSpeed.setY(vy <= -5.0 ? -5.0 : vy - gravAcc);
		windowS->playerSpeed = playerSpeed;
		dest = windowS->playerPos.add(playerSpeed);
		bool ground = playerSpeed.getY() <= 0.0001 ? collidePlayer(windowS->playerPos, &dest) : false;
		windowS->playerPos = dest;
		if (ground) {
			if (windowS->getInputValue(VK_SPACE)) {
				playerSpeed.setY(jumpPower);
			}
			else {
				playerSpeed.setY(0.0);
			}
		}

		dt2 = timeSinceEpochMillisecS();
		tickTime = (int)(dt2 - dt);
		dt = dt2;
		if (tickTime < 20) {
			std::this_thread::sleep_for(std::chrono::milliseconds(20-tickTime));
		}
	}
}