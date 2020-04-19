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

void FaceAlignment::getFaceLandmarks(const cv::Mat &image, const cv::Rect &faceArea, std::vector<cv::Point2f> &points)
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
		Point2f point(faceLandmarks.part(i).x(), faceLandmarks.part(i).y());
		points.push_back(point);
	}
}

void FaceAlignment::align(
	cv::Mat &faceAligned, const cv::Size &size,
	const cv::Mat &image,
	const std::vector<Point2f> facelandmarks)
{

	std::vector<Point2f> initialPoints;
	std::vector<Point2f> destinationPoints;

	if (facelandmarks.size() == 68)
	{
		//location of left eye left corner in input image
		initialPoints.push_back(facelandmarks[36]);

		//location of right eye right corner in input image
		initialPoints.push_back(facelandmarks[45]);
	}

	//location of left eye left corner in faceAligned image
	Point2f point1 = Point2f(0.3 * size.width, size.height / 3);

	//location of right eye right corner in faceAligned image
	Point2f point2 = Point2f(0.7 * size.width, size.height / 3);

	destinationPoints.push_back(point1);
	destinationPoints.push_back(point2);

	//calculate similarity transform
	Mat similarityMat;
	similarityTransformMat(initialPoints, destinationPoints, similarityMat);

	//define faceAligned image
	faceAligned = Mat::zeros(200, 200, CV_32FC3);

	//apply similarity transform to input image
	warpAffine(image, faceAligned, similarityMat, faceAligned.size());

	//apply similarity transform to landmark points if needed
	transform(facelandmarks, facelandmarks, similarityMat);
}

void FaceAlignment::similarityTransformMat(
		std::vector<Point2f> &initialPoints,
		std::vector<Point2f> &destinationPoints,
		cv::Mat &similarityMat)
{
	double sin60 = sin(60 * M_PI / 180.0);
	double cos60 = cos(60 * M_PI / 180.0);

	//third point is caluculated for initial points
	double initialPointsX = cos60 * (initialPoints[0].x - initialPoints[1].x) - sin60 * (initialPoints[0].y - initialPoints[1].y) + initialPoints[1].x;
	double initialPointsY = sin60 * (initialPoints[0].x - initialPoints[1].x) + cos60 * (initialPoints[0].y - initialPoints[1].y) + initialPoints[1].y;

	//third point is caluculated for destination points
	double destinationPointsX = cos60 * (destinationPoints[0].x - destinationPoints[1].x) - sin60 * (destinationPoints[0].y - destinationPoints[1].y) + destinationPoints[1].x;
	double destinationPointsY = sin60 * (destinationPoints[0].x - destinationPoints[1].x) + cos60 * (destinationPoints[0].y - destinationPoints[1].y) + destinationPoints[1].y;

	initialPoints.push_back(Point2f(initialPointsX, initialPointsY));
	destinationPoints.push_back(Point2f(destinationPointsX, destinationPointsY));

	//calculate similarity transform
	similarityMat = estimateAffinePartial2D(initialPoints, destinationPoints);
}

// void FaceAlignment::detectEyes(const Mat &faceROI, vector<Rect> &eyesDetected)
// {
// 		// Detect eyes
// 		//eyesCascade.detectMultiScale(faceROI, eyesDetected);
// }

// void FaceAlignment::getEyeCenter(const Rect &face, const Rect &eye, Point &eyeCoordinates)
// {
// 	eyeCoordinates.x = Util::getCenterOfSegment(face.x + eye.x, eye.width);
// 	eyeCoordinates.y = Util::getCenterOfSegment(face.y + eye.y, eye.height);
// }

// void FaceAlignment::getFaceCenter(const Rect &face, Point &faceCoordinates)
// {
// 	faceCoordinates.x = Util::getCenterOfSegment(face.x, face.width);
// 	faceCoordinates.y = Util::getCenterOfSegment(face.y, face.height);
// }

// double FaceAlignment::getAngleBetweenEyes(const Point &eyeA, const Point &eyeB)
// {
// 	return Util::getAngleBetweenTwoPoints(eyeA.y - eyeB.y, eyeA.x - eyeB.x);
// }

void FaceAlignment::showWindow(const Mat &img)
{
	namedWindow("Face Aligned", WINDOW_AUTOSIZE);
	imshow("Face Aligned", img);
	waitKey(0); // Wait for a keystroke in the window
}

// void FaceAlignment::drawEyes(vector<Rect> eyesDetected, const Rect &faceArea, const Mat &image)
// {
// 	for ( size_t j = 0; j < eyesDetected.size(); j++ )
// 	{
// 		Point eye_center;
// 		getEyeCenter(faceArea, eyesDetected[j], eye_center);
// 		int radius = cvRound( (eyesDetected[j].width + eyesDetected[j].height)*0.25 );
// 		circle( image, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
// 	}
// }

Mat FaceAlignment::alignFaceComplete(
		const Mat &image,
		const Rect &faceArea,
		const int height,
		const int width,
		const bool debugMode,
		const bool drawMode)
{
	//time start
	//auto start = high_resolution_clock::now();

	// if (debugMode)
	// {
	// 	cout << "debug: " << debugMode << "\n";
	// 	cout << "draw: " << drawMode << "\n";
	// }

	//Mat newImage = image.clone();					//clone tho avoid crop
	Mat alignedFace; // store final result

	//converts original image to gray scale
	//cvtColor(alignedFace, alignedFace, COLOR_BGR2GRAY);

	//vector to store face landmark points
	std::vector<Point2f> faceLandMarkPoints;
	getFaceLandmarks(image, faceArea, faceLandMarkPoints);

	Size size(width, height);
	align(alignedFace, size, image, faceLandMarkPoints);

	showWindow(alignedFace);

	//showWindow(alignedFace); // delete this

	// 1 - Detect eyes
	// std::vector<Rect> eyesDetected; // vector for eyes after cascade
	// detectEyes(alignedFace, eyesDetected); //take care moving this
	// if (debugMode || drawMode) { cout << "Eyes found: " << eyesDetected.size() << "\n"; }

	// if (eyesDetected.size() == 2)
	// {
	// 	// 2 - Get center
	// 	if (debugMode)
	// 	{
	// 		cout << "Eye One: " << eyesDetected[0] << "\n";
	// 		cout << "Eye Two: " << eyesDetected[1] << "\n";
	// 	}
	// 	Point eyeOneCenter, eyeTwoCenter;
	// 	getEyeCenter(faceArea, eyesDetected[0], eyeOneCenter);
	// 	getEyeCenter(faceArea, eyesDetected[1], eyeTwoCenter);

	// 	if(drawMode && debugMode){ drawEyes(eyesDetected, faceArea, image); }

	// 	// 3 - Get angle
	// 	double angle = getAngleBetweenEyes(eyeOneCenter, eyeTwoCenter);
	// 	if (debugMode) { cout << "Rotating: " << angle << "°\n"; }

	// 	// 4 - Rotate
	// 	Point faceCenter;
	// 	getFaceCenter(faceArea, faceCenter);
	// 	Mat rotationMatrix = getRotationMatrix2D(faceCenter, angle, 1.0);
	// 	cv::warpAffine(alignedFace, alignedFace, rotationMatrix, faceArea.size());

	// 	// 5 - Resize
	// 	if(drawMode && debugMode){
	// 		resize(image, alignedFace, Size(width, height));
	// 	}else{
	// 		resize(alignedFace, alignedFace, Size(width, height));
	// 	}

	// 	//save
	// 	//imwrite("img1.jpg", alignedFace);
	// }
	// else
	// {
	// 	cout << "Not processed\n";
	// 	if(drawMode){
	// 		drawEyes(eyesDetected, faceArea, image);
	// 	}
	// }

	// //time stop
	// auto stop = high_resolution_clock::now();
	// auto duration = duration_cast<milliseconds>(stop - start);

	// if (drawMode) { showWindow(alignedFace); }

	// if(debugMode || drawMode) {
	// 	cout << "Time: " << duration.count() << "ms\n";
	// }
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

// Mat FaceAlignment::alignFaceDebugMode(
// 		const Mat &image,
// 		const Rect &faceArea,
// 		const int height,
// 		const int width)
// {
// 	return alignFaceComplete(image, faceArea, height, width, true, false);
// }

// Mat FaceAlignment::alignFaceDrawMode(
// 		const Mat &image,
// 		const Rect &faceArea,
// 		const int height,
// 		const int width,
// 		const bool crop)
// {
// 	return alignFaceComplete(image, faceArea, height, width, crop, true);
// }