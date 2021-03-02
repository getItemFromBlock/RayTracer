#pragma once
#include <Windows.h>
#include <WinUser.h>

constexpr unsigned int keyBoardSize = 9;

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

	int* getOutputScreen();
	void setOutputScreen(int* value);

private:
	bool* inputValues; // Z Q S D SPACE L_CTRL MOUSE_L MOUSE_R
	int drawSizeX, drawSizeY;
	int* outputScreen;

	int posX, posY;
	bool captured;
};