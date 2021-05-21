/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */

#include <iostream>
#include <opencv2/video/tracking.hpp>
#include "object_recognition.h"

using namespace cv;
using namespace std;

#define LOAD_VIDEO				0		// load all video frames or only the first
#define SHOW_FEATURES			0		// show keypoints or not
#define SHOW_MATCH_KEYPOINTS	0		// show matched keypoints or not

//---------------------------------------------------------------------------functions

std::vector<cv::Mat> loadVideo(String videoName, int toogleLoading);
void drawRettangles(Mat& image, std::vector<std::vector<Point2f>> corners, std::vector<myObject> obj);
void drawRettangle(Mat& image, std::vector<Point2f>corners, myObject obj);
////////////////////////////////////////////////////////////////////////////////////////
Point2f point;
bool addRemovePt = false;

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}
////////////////////////////////////////////////////////////////////////////////////////


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
	
	/* store the corners and the points of the scene */
	std::vector<std::vector<Point2f>> scene_points = objMatcher.getScenePoints();
	std::vector<std::vector<Point2f>> scene_corners = objMatcher.getSceneCorners();
		
	/* Draw lines between the corners (the mapped object in the scene) */
	Mat detected_objects = scene.image.clone();
	drawRettangles(detected_objects, scene_corners, objects);
	namedWindow("Object detection", WINDOW_AUTOSIZE);
	imshow("Object detection", detected_objects);
	
	objMatcher.computeCenterPoints();

	/* discard the outliers */
	std::vector<std::vector<Point2f>> good_scene_points;
	for (int i = 0; i < objects.size(); i++) {
		
		std::vector<Point2f> good_scene_points_temp;
		for (int j = 0; j < scene_points[i].size(); j++) {
			Point2f distance = objMatcher.center_points[i] - scene_points[i][j];
			float distance_module = sqrt(distance.x*distance.x + distance.y*distance.y);
			if (distance_module < objMatcher.max_distance[i]) good_scene_points_temp.push_back(scene_points[i][j]);
		}
		good_scene_points.push_back(good_scene_points_temp);
	}

	/* show the keypoints without outliers */
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < good_scene_points[i].size(); j++) {
			circle(detected_objects, good_scene_points[i][j], 5, objects[i].color, -1);
		}
	}

	imshow("Object detection", detected_objects);

	/* initialize the frame and the geay frame */
	Mat old_frame, old_gray;
	old_frame = scene.image.clone();
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

	/* one vector of points for each object */
	vector<vector<Point2f>> p0;
	for (int i = 0; i < objects.size(); i++) p0.push_back(good_scene_points[i]);



	vector<Mat> mask ;
	for (int i = 0; i < objects.size(); i++) mask.push_back(Mat::zeros(old_frame.size(), old_frame.type()));
	for (int i = 0; i < objects.size(); i++) drawRettangle(mask[i], scene_corners[i], objects[i]);

	cv::VideoCapture cap("video.mov");
	TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 20, 0.03);
	
	if (cap.isOpened()) {
		for (;;) {

			Mat frame, frame_gray;
			Mat mask_new=  Mat::zeros(old_frame.size(), old_frame.type());
			cap >> frame;
			if (frame.empty())
				break;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			

			for (int i = 0; i < objects.size(); i++) {

				
				vector<Point2f>  p1;
				vector<uchar> status;
				vector<float> err;
				calcOpticalFlowPyrLK(old_gray, frame_gray, p0[i], p1, status, err, Size(31, 31), 3, criteria, 0, 0.001);

				/* select good points */
				vector<Point2f> good_new;
				vector<Point2f> good_old;
				for (uint j = 0; j < p0[i].size(); j++) {
					if (status[j] == 1) {
						good_new.push_back(p1[j]);
						good_old.push_back(p0[i][j]);
						//circle(frame, p1[j], 5, objects[i].color, -1);
					}
				}
				//cout << "old size   " << good_old.size() << "  new size  " << good_new.size() <<"-------------------------------------"<< endl;
				//vector<Mat> 
				cv::Mat affineMatrix = cv::estimateAffine2D(good_old, good_new, noArray());

				
				warpAffine(mask[i], mask_new, affineMatrix, mask[i].size());
					
				frame += mask_new;
				
			
				/* update */
				p0[i] = good_new;
				mask[i] = mask_new.clone();
			}
			
			//Mat img;
			//add(frame, mask_new, img);
			frame += mask_new;
			imshow("Frame", frame);
			int keyboard = waitKey(1);
			if (keyboard == 'q' || keyboard == 27)
				break;
			/* update of the previous frame */
			old_gray = frame_gray.clone();
			
			
		}		
	}
	else std::cout << "error 404 video not found" << std::endl;


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

void drawRettangle(Mat& image, std::vector<Point2f>corners, myObject obj) {

		for (int j = 0; j < 3; j++) line(image, corners[j], corners[j + 1], Scalar(255,0,0), 10);
		line(image, corners[3], corners[0], Scalar(255, 0, 0), 10);
	
}