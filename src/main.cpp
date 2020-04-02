#include <opencv2/core.hpp> //types
#include <opencv2/imgcodecs.hpp> // read image
#include <opencv2/objdetect.hpp> // for cascade classifier

#include <opencv2/highgui.hpp> // display image in window
#include <opencv2/imgproc.hpp> //draw

#include <iostream>
#include <math.h>

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

	for ( size_t i = 0; i < faces.size(); i++ )
	{
		//FaceAlignment::alignFace(image, faces[i], 200, 200);
		//FaceAlignment::alignFaceDebugMode(image, faces[i], 200, 200);
		FaceAlignment::alignFaceDrawMode(image, faces[i], 200, 200);
	}


	return 0;
}


// https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html

// https://hetpro-store.com/TUTORIALES/opencv-rect/ | Rect

// https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv | just use two points

// https://github.com/meefik/face-alignment/blob/master/detect.js

// https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv | crop image

// time https://www.geeksforgeeks.org/measure-execution-time-function-cpp/


/*
//get distance
				// simple euclidean distance
				//double distance = sqrt( pow(eye_center_A.x - eye_center_B.x , 2) + pow(eye_center_B.y - eye_center_B.y, 2) );
				//cout << "Distance between eyes: " << distance << "\n";
*/