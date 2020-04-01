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

```sh
#inside docker
g++ <file>.cpp `pkg-config --cflags --libs opencv4`
```
```sh
#using cmake
cmake .
make
./output <image>
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

## To Do

- [ ] Cmake
- [ ] Times