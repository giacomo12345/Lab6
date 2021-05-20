/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */

#include "object_recognition.h"

////////////////////////////////////////////////////////////////////////
//							 myObject
////////////////////////////////////////////////////////////////////////

/* constructor */
myObject::myObject() {};
myObject::myObject(Mat src) {
	cout << "Object is being created" << endl;
	Scalar clr(rand() % 255, rand() % 255, rand() % 255);
	color = clr;
	image = src;
	detector->detect(image, keypoints);
	extractor->detectAndCompute(image, Mat(), keypoints, descriptors);
}

/* show the object source image */
void myObject::showImage(){
	imshow("object image", image);
}

/* compute keypoints and descriptors */
void myObject::showKeypoints(String windowName) {
	Mat output;
	drawKeypoints(image, keypoints, output, color);
	imshow(windowName, output);
}

/* return keypoints */
std::vector<cv::KeyPoint> myObject::getKeypoints() {
	return keypoints;
}

/* return descriptors */
Mat myObject::getDescriptors() {
	return descriptors;
}


////////////////////////////////////////////////////////////////////////
//							myMatcher
////////////////////////////////////////////////////////////////////////

/* constructor */
myMatcher::myMatcher(std::vector<myObject> objects, myObject first_frame) {
	obj = objects;
	scene = first_frame;
}

/* compute the match between objects and scene image */
void myMatcher::computeMatches() {

	for (int i = 0; i < obj.size(); i++) {
		vector<DMatch> tmp_matches;
		matcher->match(obj[i].getDescriptors(), scene.getDescriptors(), tmp_matches, Mat());
		matches.push_back(tmp_matches);
	}
}