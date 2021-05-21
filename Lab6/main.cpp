/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */

#include <iostream>
#include "object_recognition.h"

using namespace cv;
using namespace std;

#define LOAD_VIDEO				0		// load all video frames or only the first
#define SHOW_FEATURES			0		// show keypoints or not
#define SHOW_MATCH_KEYPOINTS	0		// show matched keypoints or not

//---------------------------------------------------------------------------functions

std::vector<cv::Mat> loadVideo(String videoName, int toogleLoading);
void drawRettangles(Mat& image, std::vector<std::vector<Point2f>> corners, std::vector<myObject> obj);

//------------------------------------------------------------------------------------
//                               MAIN FUNCTION
//------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {

	/* upload video frames and create myObject with the first frame of the video */
	std::vector<cv::Mat> frames = loadVideo("video.mov", LOAD_VIDEO);
	myObject scene(frames[0]);

	/* upload images and create the vector of objects to be recognized inside the scene */
	std::vector<myObject> objects;
	for (int i = 1; i < 5; i++) {
		Mat img = imread(argv[i], IMREAD_COLOR);
		myObject obj(img);
		objects.push_back(obj);
		if (SHOW_FEATURES) {
			String index = to_string(i+1);
			obj.showKeypoints("index" + index);
		}
	}

	/* create the object myMatcher with the vector of object to be recognized and the scene object */
	myMatcher objMatcher(objects, scene);

	/* compute the matches */
	objMatcher.computeMatches();

	/* filter the matches according to the distance and select at least 200 features */
	objMatcher.filterMatches(1.5, 200);

	/* visualize the keypoints of object connected to the matched scene keypoints */
	if (SHOW_MATCH_KEYPOINTS) {
		std::vector<cv::Mat> img_matches = objMatcher.getImageMatched();
		for (int i = 0; i < objects.size(); i++) {
			resize(img_matches[i], img_matches[i], Size(img_matches[i].cols / 2, img_matches[i].rows / 2));
			String index = to_string(i + 1);
			imshow("Matched keypoints: image " + index, img_matches[i]);
		}
	}
	
	/* localize the objets and compute homography */
	objMatcher.computeHomography();

	/* find the corner of the object images for draw the rettangles */
	objMatcher.findCorners();

	/* compute the projection of the object image corners to the scene image */
	objMatcher.computeProjection();
	
	/* store the corners and the points of the scene image */
	std::vector<std::vector<Point2f>> scene_points = objMatcher.getScenePoints();
	std::vector<std::vector<Point2f>> scene_corners = objMatcher.getSceneCorners();

	
	/* Draw lines between the corners (the mapped object in the scene) */
	Mat detected_objects = scene.image.clone();
	drawRettangles(detected_objects, scene_corners, objects);
	namedWindow("Object detection", WINDOW_AUTOSIZE);
	imshow("Object detection", detected_objects);
	
	cout << "END" << std::endl;
	waitKey(0);
	return 0;
}
//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------
std::vector<cv::Mat> loadVideo(String videoName, int toogleLoading = LOAD_VIDEO) {

	cout << "frames uploading";
	std::vector<cv::Mat> frames_to_load;
	cv::VideoCapture cap(videoName);
	if (cap.isOpened()) {
		int loader = 0;
		for (;;) {
			cv::Mat frame;
			cap >> frame;
			if (!cap.read(frame)) break;
			frames_to_load.push_back(frame);
			if (loader % 110 == 0) cout << ".";
			loader++;
			if (!LOAD_VIDEO) break;
			/* namedWindow("video", WINDOW_FREERATIO);  imshow("video", frame); waitKey(1); */
		}
	}
	else std::cout << "error 404 video not found" << std::endl;
	std::cout << std::endl;
	cout << "number frames found: " << frames_to_load.size() << std::endl;
	cap.release();  //destroyWindow("video");
	
	return frames_to_load;
}

void drawRettangles(Mat& image, std::vector<std::vector<Point2f>> corners, std::vector<myObject> obj) {
	for (int i = 0; i < obj.size(); i++) {
		for (int j = 0; j < 3; j++) line(image, corners[i][j], corners[i][j + 1], obj[i].color, 4);
		line(image, corners[i][3], corners[i][0] ,obj[i].color, 4);
	}
}