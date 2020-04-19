#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // for rotate, save, change color
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp> // display image in window

#include <dlib/opencv.h> //cvimage

#include <dlib/image_processing.h>											 //shape_predictor
#include <dlib/image_processing/frontal_face_detector.h> // frontal face

#include <chrono> //for time meassure
#include <iostream>

#include "align.hpp"
#include "util.hpp"

using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace dlib;

FaceAlignment::FaceAlignment()
{
	// file which is a pre-trained cascade of regression tree implemented using
	// "One Millisecond face alignment with an ensemble of regression trees"
	faceLandmarkModel = "/root/workspace/models/shape_predictor_68_face_landmarks.dat";
	loadModel();
}

FaceAlignment::FaceAlignment(const string path)
{
	faceLandmarkModel = path;
	loadModel();
}

void FaceAlignment::loadModel()
{
	try
	{
		// https://github.com/davisking/dlib/blob/master/dlib/serialize.h
		deserialize(faceLandmarkModel) >> landmarkDetector;
	}
	catch (const dlib::serialization_error &e)
	{
		cerr << e.what() << '\n';
		cout << "Please check your route or file is corrupted!\n";
	}
}

void FaceAlignment::getFaceLandmarks(const cv::Mat &image, const cv::Rect &faceArea, std::vector<cv::Point> &points)
{
	//convert image to dlib image format
	//http://dlib.net/imaging.html
	cv_image<bgr_pixel> dlibImage(image);

	//http://dlib.net/python/index.html#dlib.rectangle
	dlib::rectangle face;

	// convert Rect from opencv to rectangle dblib
	Util::cvRecttodlibRectangle(faceArea, face);

	//get face landmarks
	//http://dlib.net/python/index.html#dlib.full_object_detection
	full_object_detection faceLandmarks = landmarkDetector(dlibImage, face);

	// shape predictor stores the 68 landmark points in dlib::point from
	//convert full_object_detection to vector form
	for (int i = 0; i < faceLandmarks.num_parts(); i++)
	{
		Point point(faceLandmarks.part(i).x(), faceLandmarks.part(i).y());
		// cout << point.x  << '\n';
		// cout << point.y << '\n';
		points.push_back(point);
	}
}

void FaceAlignment::getEyesCoordinates(
		const std::vector<cv::Point> &facelandmarks,
		cv::Point &leftEye,
		cv::Point &rightEye)
{
	// using 68 landmark
	if (facelandmarks.size() == 68)
	{
		//location of left eye left corner in input image
		leftEye = facelandmarks[36];

		//location of right eye right corner in input image
		rightEye = facelandmarks[45];
	}
}

void FaceAlignment::getFaceCenter(const Rect &face, Point &faceCoordinates)
{
	faceCoordinates.x = Util::getCenterOfSegment(face.x, face.width);
	faceCoordinates.y = Util::getCenterOfSegment(face.y, face.height);
}

double FaceAlignment::getAngleBetweenEyes(const Point &eyeA, const Point &eyeB)
{
	// Util::getAngleBetweenTwoPoints(eyeA.y - eyeB.y, eyeA.x - eyeB.x);
	return Util::getAngleBetweenTwoPoints(eyeB.y - eyeA.y, eyeB.x - eyeA.x);
}

void FaceAlignment::showWindow(const Mat &img)
{
	namedWindow("Face Aligned", WINDOW_AUTOSIZE);
	imshow("Face Aligned", img);
	waitKey(0); // Wait for a keystroke in the window
}

Mat FaceAlignment::alignFaceComplete(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width,
		const bool debugMode,
		const bool drawMode)
{
	//time start
	auto start = high_resolution_clock::now();

	Mat alignedFace = image(faceArea); // store final result

	//converts original image to gray scale
	//cvtColor(alignedFace, alignedFace, COLOR_BGR2GRAY);

	// 1 - Get landmarks
	//vector to store face landmark points
	std::vector<Point> faceLandMarkPoints;
	getFaceLandmarks(image, faceArea, faceLandMarkPoints);

	// 2 - Detect eyes
	Point leftEye, rightEye;
	getEyesCoordinates(faceLandMarkPoints, leftEye, rightEye);

	// 3 - Get angle
	double angle = getAngleBetweenEyes(leftEye, rightEye);

	if (debugMode)
	{
		cout << "Angle : " << angle << '\n';
	}

	// 4 - Rotate
	Point faceCenter;
	getFaceCenter(faceArea, faceCenter);
	Mat rotationMatrix = getRotationMatrix2D(faceCenter, angle, 1.0);
	warpAffine(alignedFace, alignedFace, rotationMatrix, faceArea.size());

	// 5 - Resize
	if (drawMode && debugMode)
	{
		resize(image, alignedFace, Size(width, height));
	}
	else
	{
		resize(alignedFace, alignedFace, Size(width, height));
	}

	//time stop
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	if (debugMode || drawMode)
	{
		cout << "Time: " << duration.count() << "ms\n";
	}

	if (drawMode)
	{
		showWindow(alignedFace);
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