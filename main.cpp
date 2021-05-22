/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */

#include <iostream>
#include <opencv2/video/tracking.hpp>
#include "object_recognition.h"

using namespace cv;
using namespace std;

#define LOAD_VIDEO				1		// load all video frames or only the first
#define SHOW_FEATURES			1		// show keypoints or not
#define SHOW_MATCH_KEYPOINTS	1		// show matched keypoints or not

//---------------------------------------------------------------------------functions

std::vector<cv::Mat> loadVideo(String videoName, int toogleLoading);
void drawRettangles(Mat& image, std::vector<std::vector<Point2f>> corners, std::vector<myObject> obj);
void drawRettangle(Mat& image, std::vector<Point2f>corners, myObject obj);

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
	
	//////////////////////////////////////////////////////////////////////////////////////////CLASSI DA QUI IN POI
	/* store the corners and the points of the scene */
	std::vector<std::vector<Point2f>> scene_points = objMatcher.getScenePoints();
	std::vector<std::vector<Point2f>> scene_corners = objMatcher.getSceneCorners();
		
	/* Draw lines between the corners (the mapped object in the scene) */
	Mat detected_objects = scene.image.clone();
	drawRettangles(detected_objects, scene_corners, objects);
	namedWindow("Object detection", WINDOW_AUTOSIZE);
	imshow("Object detection", detected_objects);

	/* discard the outliers */
	objMatcher.computeCenterAndDistance();
	objMatcher.computeGoodScenePoints();

	/* show the keypoints without outliers */						
	std::vector<std::vector<Point2f>> good_scene_points = objMatcher.getGoodScenePoints();
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < good_scene_points[i].size(); j++) {
			circle(detected_objects, good_scene_points[i][j], 5, objects[i].color, -1);
		}
	}

	imshow("Object detection", detected_objects);

	/*---------------------------------------------------------------------------------*/
	//									TRACKING									   //
	/*---------------------------------------------------------------------------------*/

	/* initialize the first frame and the gray frame */
	Mat old_frame, old_gray;
	old_frame = scene.image.clone();
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

	TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 20, 0.03);
	
	/* for every video frames */
	for (int frm = 0; frm < frames.size(); frm++) {
		Mat frame, frame_gray;
		frame = frames[frm];
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

		/* for every object to track */
		for (int i = 0; i < objects.size(); i++) {

			vector<Point2f> tracked_points;
			vector<uchar> status;
			vector<float> err;

			/* track good_scene_points and find tracked_points*/
			calcOpticalFlowPyrLK(old_gray, frame_gray, good_scene_points[i], tracked_points, status, err, Size(31, 31), 3, criteria, 0, 0.001);

			/* select good points */
			vector<Point2f> good_new_points;
			vector<Point2f> good_old_points;
			for (uint j = 0; j < good_scene_points[i].size(); j++) {
				if (status[j] == 1) {
					good_new_points.push_back(tracked_points[j]);
					good_old_points.push_back(good_scene_points[i][j]);
				}
			}

			/* estimation of the affine matrix between two consecutive frames */
			cv::Mat_<float> affineMatrix = cv::estimateAffine2D(good_old_points, good_new_points, noArray());

			/* compute the new position of the corners according to the affine transformation */
			for (int corner = 0; corner < scene_corners[i].size(); corner++) {

				Mat_<float> old_corner(3, 1);
				old_corner << scene_corners[i][corner].x, scene_corners[i][corner].y, 1.0;
				Mat_<float> new_corner = affineMatrix * old_corner;

				/* update corners */
				scene_corners[i][corner] = Point2f(new_corner(0), new_corner(1));
			}

			/* draw the rettangle */
			drawRettangle(frame, scene_corners[i], objects[i]);
			
			/* update scene points */
			good_scene_points[i] = good_new_points;
		}

		imshow("Video - object tracking", frame);
		waitKey(1);

		/* update of the previous frame */
		old_gray = frame_gray.clone();
	
	}
	


	cout << "----END----" << std::endl;
	waitKey(0);
	return 0;
}
//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------

/* upload the video frames */
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
		}
	}
	else std::cout << "error - video not found" << std::endl;
	std::cout << std::endl;
	cout << "number frames found: " << frames_to_load.size() << std::endl;
	cap.release(); 

	return frames_to_load;
}

/* draw rettangles of multiple objects */
void drawRettangles(Mat& image, std::vector<std::vector<Point2f>> corners, std::vector<myObject> obj) {
	for (int i = 0; i < obj.size(); i++) {
		for (int j = 0; j < 3; j++) line(image, corners[i][j], corners[i][j + 1], obj[i].color, 4);
		line(image, corners[i][3], corners[i][0] ,obj[i].color, 4);
	}
}

/* draw rettangle of a single object */
void drawRettangle(Mat& image, std::vector<Point2f>corners, myObject obj) {

		for (int j = 0; j < 3; j++) line(image, corners[j], corners[j + 1],obj.color, 3);
		line(image, corners[3], corners[0], obj.color, 3);
	
}