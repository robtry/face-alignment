#include <opencv2/core.hpp> //types
#include <opencv2/imgcodecs.hpp> // read image

#include <iostream>

#include "faceDetection/module1.hpp"
#include "faceAlignment/align.hpp"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	if (argc != 2) { cout << " There is no image to load \n"; return -1; }

	// Image from input
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) { cout << "Could not open or find the image \n"; return -1; }

	// Detect Faces
	vector<Rect> faces;
	Module1::detectFaces(faces, image);

	//Aling face
	FaceAlignment aling;

	for ( size_t i = 0; i < faces.size(); i++ )
	{
		//aling.alignFace(image, faces[i], 200, 200);
		aling.alignFaceDebugMode(image, faces[i], 200, 200);
		//aling.alignFaceDrawMode(image, faces[i], 200, 200, true);
	}

	return 0;
}
