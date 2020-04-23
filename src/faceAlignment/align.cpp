#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // display image in window
#include <dlib/opencv.h> //cvimage

#include <chrono> //for time meassure
#include <iostream>

#include "align.hpp"

using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace dlib;

FaceAlignment::FaceAlignment()
{
	// file which is a pre-trained cascade of regression tree implemented using
	// "One Millisecond face alignment with an ensemble of regression trees"
	faceLandmarkModel = "../models/shape_predictor_5_face_landmarks.dat";
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

void FaceAlignment::cvRecttodlibRectangle(const Rect &r, dlib::rectangle &rec)
{
	rec = dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}


Mat FaceAlignment::alignFaceComplete(
		const Mat &image,
		const Rect &faceArea,
		const int size,
		const bool debugMode,
		const bool drawMode)
{
	//time start
	auto start = high_resolution_clock::now();

	// convert to dlib
	cv_image<bgr_pixel> dlibImage(image);
	dlib::rectangle face;
	cvRecttodlibRectangle(faceArea, face);

	//1 - Detect Lankmarks
	full_object_detection faceLandmarks = landmarkDetector(dlibImage, face);

	//2 - Aling
	matrix<bgr_pixel> face_chips; // https://github.com/davisking/dlib/issues/1648
	// https://stackoverflow.com/questions/29419274/region-of-interest-extraction-using-dlib
	// http://dlib.net/dlib/image_transforms/interpolation_abstract.h.html#extract_image_chips
	// http://dlib.net/dlib/image_transforms/interpolation_abstract.h.html#get_face_chip_details

	extract_image_chip(dlibImage, get_face_chip_details(faceLandmarks, size), face_chips);

	// 3- Change to Mat
	Mat matAlignedFace = toMat(face_chips);

	//time stop
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	if (debugMode)
	{
		cout << "Time: " << duration.count() << "ms\n";
	}

	if (drawMode)
	{
		namedWindow("Face Aligned", WINDOW_AUTOSIZE);
		imshow("Face Aligned", matAlignedFace);
		waitKey(0); // Wait for a keystroke in the window
	}

	return matAlignedFace;
}

Mat FaceAlignment::alignFace(
		const Mat &image,
		const Rect &faceArea,
		const int size)
{
	return alignFaceComplete(image, faceArea, size, false, false);
}

Mat FaceAlignment::alignFaceDebugMode(
		const Mat &image,
		const Rect &faceArea,
		const int size,
		const bool draw)
{
	return alignFaceComplete(image, faceArea, size, true, draw);
}
