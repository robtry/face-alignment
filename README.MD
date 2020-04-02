# Face Alignment

## Start:

```sh
# Get the image
docker pull https://hub.docker.com/repository/docker/robtry/opencv-420

# Enable xhost
xhost +

# Linux
docker run --privileged --device /dev/video0:/dev/video0 -v $(pwd):/root/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8888:8888 -it robtry/opencv-420

# Mac
sudo docker run -it -v $(pwd):/root/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=docker.for.mac.host.internal:0  -p 5000:5000 -p 8888:8888 robtry/opencv-420
```

## Compile

You can follow one the next options:

```sh
#inside docker
g++ <file>.cpp `pkg-config --cflags --libs opencv4`
```

```sh
#using cmake
cmake .
make
./exe <image>
```

```sh
#using a simple script
./test/premake.sh
```


## Autocomplete in vscode:

```sh
# from docker to this workspace
cp -r /usr/include/opencv4/ .
```

Then add to configuration file

```js
{
	"configurations": [
		{
			"includePath": [
				"${workspaceFolder}/**",
				"${workspaceFolder}/opencv4/"
			],
		}
}
```

## How to use 

Just copy the foler `src/faceAlignment` to your project. Then in you `main.cpp`

```c++
#include "faceAlignment/align.hpp"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	//something important here
	//...


	for ( size_t i = 0; i < faces.size(); i++ )
	{
		// just use one of this
		FaceAlignment::alignFace(image, faces[i], 40, 40);
		FaceAlignment::alignFaceDebugMode(image, faces[i], 40, 40);
		FaceAlignment::alignFaceDrawMode(image, faces[i], 40, 40, true);
	}

	return 0;
}
```

## To Do

- [ ] Review 1