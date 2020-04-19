// http://dlib.net/python/index.html
// http://dlib.net/ml.html
// http://dlib.net/face_landmark_detection_ex.cpp.html


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // for rotate, save
#include <opencv2/highgui.hpp> // display image in window

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
	loadModelAndStartDetector();
}

FaceAlignment::FaceAlignment(const string path)
{
	faceLandmarkModel = path;
	loadModelAndStartDetector();
}

void FaceAlignment::loadModelAndStartDetector()
{
	// to get bounding boxes for each face in an image
	// http://dlib.net/python/index.html#dlib.get_frontal_face_detector
	faceDetector = get_frontal_face_detector();

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

void getFaceLandmarks(const cv::Mat &image, std::vector<cv::Point2f> &points, shape_predictor landmarkDetector){
	//get face landmarks
		full_object_detection faceLandmarks = landmarkDetector(dlibImage, firstFace);

		//convert full_object_detection to vector form
		for (int i = 0; i < faceLandmarks.num_parts(); i++)
		{
			Point2f point(faceLandmarks.part(i).x(), faceLandmarks.part(i).y());
			points.push_back(point);
		}
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

// void FaceAlignment::showWindow(const Mat &img)
// {
// 	namedWindow("Face Aligned", WINDOW_AUTOSIZE);
// 	imshow("Face Aligned", img);
// 	waitKey(0); // Wait for a keystroke in the window
// }

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

	Mat newImage = image.clone(); //clone tho avoid crop
	Mat alignedFace = newImage(faceArea); // store final result

	//converts original image to gray scale
	cvtColor(alignedFace, alignedFace, COLOR_BGR2GRAY);

	//vector to store face landmark points
	std::vector<Point2f> points;
	getFaceLandmarks(alignedFace, points, landmarkDetector);
	cout << points.size() << '\n';

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