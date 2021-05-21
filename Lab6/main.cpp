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

	objMatcher.findCorners();
	objMatcher.computeProjection();
	////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// CLASSI FINO A QUI /////////////////////////////////////
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
	//std::vector<std::vector<DMatch>> good_matches = objMatcher.good_matches;


	/* vectors for store the results */
	//std::vector<std::vector<Point2f>> obj_points;
	//std::vector<std::vector<Point2f>> scene_points;
	//std::vector<std::vector<Point2f>> obj_corners;
	std::vector<std::vector<Point2f>> scene_corners = objMatcher.getSceneCorners();
	//std::vector<Mat> H;
	//std::vector< std::vector<float>> coord;// X min & max , Y min & max

	//for (int i = 0; i < objects.size(); i++) {
	//	std::vector<float > temp_coord;
	//	temp_coord.push_back(objects[i].getKeypoints()[0].pt.x);
	//	temp_coord.push_back(objects[i].getKeypoints()[0].pt.x);
	//	temp_coord.push_back(objects[i].getKeypoints()[0].pt.y);
	//	temp_coord.push_back(objects[i].getKeypoints()[0].pt.y);
	//	coord.push_back(temp_coord);
	//}

	//for (int j = 0; j < objects.size(); j++) {
	//	for (int i = 0; i < objects[j].getKeypoints().size(); i++) {

	//		float x = objects[j].getKeypoints()[i].pt.x;
	//		float y = objects[j].getKeypoints()[i].pt.y;

	//		if (x < coord[j][0]) coord[j][0] = x;
	//		else if (x > coord[j][1]) coord[j][1] = x;

	//		if (y < coord[j][2]) coord[j][2] = y;
	//		else if (y > coord[j][3]) coord[j][3] = y;
	//	}
	//}

	/* localize object - compute homography - get the corners from the object to be detected */
	for (int i = 0; i < objects.size(); i++) {

		///* temp elementes - will be stored in the outside vectors */
		//std::vector<Point2f> temp_obj_points;
		//std::vector<Point2f> temp_scene_points;
		//std::vector<Point2f> temp_obj_corners(4);
		//std::vector<Point2f> temp_scene_corners = objMatcher.getSceneCorners();

		///* Localize the object */
		//for (size_t j = 0; j < good_matches[i].size(); j++) {
		//	temp_obj_points.push_back(objects[i].getKeypoints()[good_matches[i][j].queryIdx].pt);
		//	temp_scene_points.push_back(scene.getKeypoints()[good_matches[i][j].trainIdx].pt);
		//}


		//
		//

		///* compute homography */
		//Mat temp_H = findHomography(temp_obj_points, temp_scene_points, RANSAC);

		///* Get the corners from the object to be detected */
		//temp_obj_corners[0] = Point2f(coord[i][0], coord[i][2]);
		//temp_obj_corners[1] = Point2f(coord[i][0], coord[i][3]);
		//temp_obj_corners[2] = Point2f(coord[i][1], coord[i][3]);
		//temp_obj_corners[3] = Point2f(coord[i][1], coord[i][2]);

		//temp_obj_corners[0] = Point2f(0, 0);
		//temp_obj_corners[1] = Point2f((float)objects[i].image.cols, 0);
		//temp_obj_corners[2] = Point2f((float)objects[i].image.cols, (float)objects[i].image.rows);
		//temp_obj_corners[3] = Point2f(0, (float)objects[i].image.rows);

		//perspectiveTransform(objMatcher.obj_corners[i], temp_scene_corners, objMatcher.H[i]);

		///* store the "temp" elements in the outside vectors */
		//obj_points.push_back(temp_obj_corners);
		//scene_points.push_back(temp_scene_corners);
		//obj_corners.push_back(temp_obj_corners);
		//scene_corners.push_back(temp_scene_corners);
		//H.push_back(temp_H);
	}


	/* ------------------------SHOW THE RESULT---------------------------------- */
	Mat result = scene.image.clone();

	/* Draw lines between the corners (the mapped object in the scene) */
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < 3; j++) line(result, scene_corners[i][j], scene_corners[i][j + 1], objects[i].color, 4);
		line(result, scene_corners[i][3], scene_corners[i][0], objects[i].color, 4);
	}

	namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
	//resize(img_matches[0], img_matches[0], Size(img_matches[0].cols / 2, img_matches[0].rows / 2));
	imshow("Good Matches & Object detection", result);

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
