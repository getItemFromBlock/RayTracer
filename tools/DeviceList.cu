template<typename T>
class DeviceList
{
private:
	T* objList;

	size_t capacity;
	size_t length;
	__device__ void expand() {
		capacity *= 2;
		T* tempObj = new T[capacity];

		for (size_t i = 0; i < length; i++) {
			tempObj[i] = objList[i];
		}
		delete[] objList;
		objList = tempObj;
	}
public:
	__device__ explicit DeviceList() : length(0), capacity(16) {
		objList = new T[capacity];
	}
	__device__ T operator[] (int index) {
		return objList[index];//*(begin+index)
	}
	__device__ T begin() {
		return objList[0];
	}
	__device__ T end() {
		return objList[length-1];
	}
	__device__ ~DeviceList()
	{
		delete[] objList;
		objList = nullptr;
	}

	__device__ void add(T t) {

		if (length >= capacity) {
			expand();
		}
		objList[length] = t;
		length++;
	}
	__device__ T pop() {
		T endElement = end();
		objList[length - 1] = 0;
		length--;
		return endElement;
	}

	__device__ T at(int index) {
		if (index < length)
		{
			return objList[index];
		}
		return end();
	}

	__device__ void set(int index, T t) {
		if (index < length)
		{
			if (index < 0) {
				index = length + index;
			}
			objList[index] = t;
			return;
		}
		return;
	}

	__device__ size_t getSize() {
		return length;
	}
};