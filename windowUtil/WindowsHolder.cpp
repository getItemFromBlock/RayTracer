#include "WindowsHolder.h"

WindowsHolder::WindowsHolder(HWND param): windowH(param)
{
	inputValues = (bool*)malloc(keyBoardSize*sizeof(bool));
	for (unsigned int i = 0; i < keyBoardSize; i++) {
		inputValues[i] = false;
	}
}

WindowsHolder::~WindowsHolder()
{
	free(inputValues);
}

void WindowsHolder::setInputValue(char input, bool state)
{
	switch (input)
	{
	case 'Z':
	{
		inputValues[0] = state;
		break;
	}
	case 'Q':
	{
		inputValues[1] = state;
		break;
	}
	case 'S':
	{
		inputValues[2] = state;
		break;
	}
	case 'D':
	{
		inputValues[3] = state;
		break;
	}
	case VK_SPACE:
	{
		inputValues[4] = state;
		break;
	}
	case VK_CONTROL:
	{
		inputValues[5] = state;
		break;
	}
	case VK_LBUTTON:
	{
		inputValues[6] = state;
		break;
	}
	case VK_RBUTTON:
	{
		inputValues[7] = state;
		break;
	}
	case VK_ESCAPE:
	{
		inputValues[8] = state;
		break;
	}

	return;
	}
}

void WindowsHolder::setMousePosInput(int x, int y)
{
	posX = x;
	posY = y;
}

bool WindowsHolder::getInputValue(char input)
{
	bool result = false;
	switch (input)
	{
	case 'Z':
	{
		result = inputValues[0];
		break;
	}
	case 'Q':
	{
		result = inputValues[1];
		break;
	}
	case 'S':
	{
		result = inputValues[2];
		break;
	}
	case 'D':
	{
		result = inputValues[3];
		break;
	}
	case VK_SPACE:
	{
		result = inputValues[4];
		break;
	}
	case VK_CONTROL:
	{
		result = inputValues[5];
		break;
	}
	case VK_LBUTTON:
	{
		result = inputValues[6];
		break;
	}
	case VK_RBUTTON:
	{
		result = inputValues[7];
		break;
	}
	case VK_ESCAPE:
	{
		result = inputValues[8];
		break;
	}

	}
	return result;
}

int WindowsHolder::getMousePosX()
{
	return posX;
}

int WindowsHolder::getMousePosY()
{
	return posY;
}

void WindowsHolder::setCaptured(bool value)
{
	captured = value;
}

bool WindowsHolder::isCaptured()
{
	return captured;
}

void WindowsHolder::setDrawSize(int width, int height)
{
	drawSizeX = width;
	drawSizeY = height;
}

int WindowsHolder::getDrawSizeX()
{
	return drawSizeX;
}

int WindowsHolder::getDrawSizeY()
{
	return drawSizeY;
}

int* WindowsHolder::getOutputScreen()
{
	return outputScreen;
}

void WindowsHolder::setOutputScreen(int* value)
{
	int* tempScreen = outputScreen;
	outputScreen = value;
	free(tempScreen);
}