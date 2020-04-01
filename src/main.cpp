#include <opencv2/core.hpp> //types
#include <opencv2/imgcodecs.hpp> // read image
#include <opencv2/objdetect.hpp> // for cascade classifier

#include <opencv2/highgui.hpp> // display image in window
#include <opencv2/imgproc.hpp> //draw

#include <iostream>
#include <math.h>

#include "Module1.hpp"

using namespace cv;
using namespace std;

// Path to the model
//const string FACEMODEL = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
//const string EYESMODEL = "/opt/opencv/data/haarcascades/haarcascade_eye.xml";

int main(int argc, char **argv)
{
	if (argc != 2) { cout << " There is no image to load \n"; return -1; }

	// Image from input
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) { cout << "Could not open or find the image \n"; return -1; }

	

	return 0;
}


// https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html

// https://hetpro-store.com/TUTORIALES/opencv-rect/ | Rect

// https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv | just use two points

// https://github.com/meefik/face-alignment/blob/master/detect.js

// https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv | crop image


/*
//get distance
				// simple euclidean distance
				//double distance = sqrt( pow(eye_center_A.x - eye_center_B.x , 2) + pow(eye_center_B.y - eye_center_B.y, 2) );
				//cout << "Distance between eyes: " << distance << "\n";
*/