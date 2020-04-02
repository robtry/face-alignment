#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp> // for cascade classifier
#include <opencv2/imgproc.hpp>	 // for rotate, save
#include <opencv2/highgui.hpp>	 // display image in window
#include <chrono> //for time meassure
#include <iostream>

#include "align.hpp"
#include "util.hpp"

using namespace std::chrono; 
using namespace std;
using namespace cv;

// Paths to the model
const string FaceAlignment::EYESMODEL = "/opt/opencv/data/haarcascades/haarcascade_eye.xml";

void FaceAlignment::detectEyes(const Mat &faceROI, vector<Rect> &eyesDetected)
{
	// Load cascade
	CascadeClassifier eyesCascade;
	if (!eyesCascade.load(FaceAlignment::EYESMODEL))
	{
		cout << "Error loading eyes cascade\n"
				 << "Check path in \"align.cpp\"\n";
	}
	else
	{
		// Detect eyes
		eyesCascade.detectMultiScale(faceROI, eyesDetected);
	}
}

void FaceAlignment::getEyeCenter(const Rect &face, const Rect &eye, Point &eyeCoordinates)
{
	eyeCoordinates.x = Util::getCenterOfSegment(face.x + eye.x, eye.width);
	eyeCoordinates.y = Util::getCenterOfSegment(face.y + eye.y, eye.height);
}

void FaceAlignment::getFaceCenter(const Rect &face, Point &faceCoordinates)
{
	faceCoordinates.x = Util::getCenterOfSegment(face.x, face.width);
	faceCoordinates.y = Util::getCenterOfSegment(face.y, face.height);
}

double FaceAlignment::getAngleBetweenEyes(const Point &eyeA, const Point &eyeB)
{
	return Util::getAngleBetweenTwoPoints(eyeA.y - eyeB.y, eyeA.x - eyeB.x);
}

void FaceAlignment::showWindow(const Mat &img)
{
	namedWindow("Face Aligned", WINDOW_AUTOSIZE);
	imshow("Face Aligned", img);
	waitKey(0); // Wait for a keystroke in the window
}

void FaceAlignment::drawEyes(vector<Rect> eyesDetected, const Rect &faceArea, const Mat &image)
{
	for ( size_t j = 0; j < eyesDetected.size(); j++ )
	{
		Point eye_center;
		getEyeCenter(faceArea, eyesDetected[j], eye_center);
		int radius = cvRound( (eyesDetected[j].width + eyesDetected[j].height)*0.25 );
		circle( image, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
	}
}

Mat FaceAlignment::alignFaceComplete(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width,
		const bool debugMode,
		const bool drawMode)
{
	// https://www.geeksforgeeks.org/measure-execution-time-function-cpp/ | meassure time
	//time start
	auto start = high_resolution_clock::now();

	if(debugMode){
		cout << "debug: " << debugMode << "\n";
		cout << "draw: " << drawMode << "\n";
	}

	Mat alignedFace = image(faceArea); // store final result

	// 1 - Detect eyes
	vector<Rect> eyesDetected; // vector for eyes after cascade
	detectEyes(image(faceArea), eyesDetected); //take care moving this
	if (debugMode || drawMode) { cout << "Eyes found: " << eyesDetected.size() << "\n"; }

	if (eyesDetected.size() == 2)
	{
		// 2 - Get center
		if (debugMode)
		{
			cout << "Eye One: " << eyesDetected[0] << "\n";
			cout << "Eye Two: " << eyesDetected[1] << "\n";
		}
		Point eyeOneCenter, eyeTwoCenter;
		getEyeCenter(faceArea, eyesDetected[0], eyeOneCenter);
		getEyeCenter(faceArea, eyesDetected[1], eyeTwoCenter);

		if(drawMode && debugMode){ drawEyes(eyesDetected, faceArea, image); }

		// 3 - Get angle
		double angle = getAngleBetweenEyes(eyeOneCenter, eyeTwoCenter);
		if (debugMode) { cout << "Rotating: " << angle << "°\n"; }

		// 4 - Rotate
		Point faceCenter;
		getFaceCenter(faceArea, faceCenter);
		Mat rotationMatrix = getRotationMatrix2D(faceCenter, angle, 1.0);	
		warpAffine(alignedFace, alignedFace, rotationMatrix, faceArea.size());

		// 5 - Resize
		if(drawMode && debugMode){
			resize(image, alignedFace, Size(width, height));
		}else{
			resize(alignedFace, alignedFace, Size(width, height));
		}

		//save 
		//imwrite("img1.jpg", alignedFace);
	}
	else
	{
		cout << "Not processed\n";
		if(drawMode){
			drawEyes(eyesDetected, faceArea, image);
		}
	}

	//time stop
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start); 

	if (drawMode) { showWindow(alignedFace); }

	if(debugMode || drawMode) { 
		cout << "Time: " << duration.count() << "ms\n";
	}
	return alignedFace;
}

Mat FaceAlignment::alignFace(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width)
{
	return alignFaceComplete(image, faceArea, height, width, false, false);
}

Mat FaceAlignment::alignFaceDebugMode(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width)
{
	return alignFaceComplete(image, faceArea, height, width, true, false);
}

Mat FaceAlignment::alignFaceDrawMode(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width,
		const bool crop)
{
	return alignFaceComplete(image, faceArea, height, width, crop, true);
}