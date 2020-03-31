#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> // read image
//#include <opencv2/highgui.hpp> // display image in window
#include <opencv2/objdetect.hpp> // for cascade classifier

#include <iostream>

using namespace cv;
using namespace std;

// Path to the model
const string FACEMODEL = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
const string EYESMODEL = "/opt/opencv/data/haarcascades/haarcascade_eye.xml";

int main(int argc, char **argv)
{
	// There is no arg in the excecution
	if (argc != 2) { cout << " There is no image to load \n"; return -1; }

	// Image from input
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) { cout << "Could not open or find the image \n"; return -1; }

	// PreDefined trained XML classifiers with facial features
	CascadeClassifier face_cascade, eyes_cascade;

	// Load classifiers
	if (!face_cascade.load(FACEMODEL)) { cout << "Error loading face cascade\n"; return -1; }
	if (!eyes_cascade.load(EYESMODEL)) { cout << "Error loading eyes cascade\n"; return -1; }


	// Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale( image, faces );

	cout << "Faces found: " << faces.size() << "\n\n";
	for ( size_t i = 0; i < faces.size(); i++ )
	{
		cout << "Face[" << i + 1 << "] | " << "ROI: " << faces[i] << "\n";

		Mat faceROI = image( faces[i] );
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		cout << "Eyes found: " << eyes.size() << "\n";
		if(eyes.size() == 2) {
			for ( size_t i = 0; i < eyes.size(); i++ )
			{
				cout << "Eye: " << eyes[i] << "\n";
			}
			cout << "\n";
		} else {
			cout << "No se procesa" << "\n";
		}
	}

	//show images
	// namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
	// imshow( "Display window", image ); // Show our image inside it.
	// waitKey(0); // Wait for a keystroke in the window
	return 0;
}


// https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html

// https://hetpro-store.com/TUTORIALES/opencv-rect/ | Rect