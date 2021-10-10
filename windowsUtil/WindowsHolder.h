#pragma once
#include <Windows.h>
#include <WinUser.h>
#include "../tools/Vector3D.cuh"

constexpr unsigned int keyBoardSize = 10;



class WindowsHolder {

public:

	WindowsHolder(HWND param);
	~WindowsHolder();

	void setInputValue(char input, bool state);
	bool getInputValue(char input);
	void setMousePosInput(int x, int y);
	void setDrawSize(int width, int height);
	void setCaptured(bool value);
	bool isCaptured();
	int getMousePosX();
	int getMousePosY();
	
	const HWND windowH;
	bool exitWindow = false;
	bool sizing = false;
	int getDrawSizeX();
	int getDrawSizeY();

	Vector3D playerPos = Vector3D(1, 5.0, 0);
	Vector3D playerDir = Vector3D();
	Vector3D playerSpeed = Vector3D();

	float rotX = 0.0;
	float rotY = 0.0;

	int* getOutputScreen();
	void setOutputScreen(int* value);

private:
	bool* inputValues; // Z Q S D SPACE L_CTRL MOUSE_L MOUSE_R
	int drawSizeX = 0, drawSizeY = 0;
	int* outputScreen = nullptr;

	int posX = 0, posY = 0;
	bool captured = false;
};