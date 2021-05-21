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

	/* get the distance from the center point and each keypoint and discard if is bigger than max distance */
	std::vector<std::vector<Point2f>> good_scene_points;
	for (int i = 0; i < objects.size(); i++) {
		
		std::vector<Point2f> good_scene_points_temp;
		for (int j = 0; j < scene_points[i].size(); j++) {
			Point2f distance = objMatcher.center_points[i] - scene_points[i][j];
			float distance_module = sqrt(distance.x*distance.x + distance.y*distance.y);
			if (distance_module < objMatcher.max_distance[i]) good_scene_points_temp.push_back(scene_points[i][j]);
		}

		good_scene_points.push_back(good_scene_points_temp);

		cout << "oushato" << i << endl;

		
	}
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < good_scene_points[i].size(); j++) {

			circle(detected_objects, good_scene_points[i][j], 5, objects[i].color, -1);
		}
	}
	imshow("Object detection", detected_objects);


	
	const int MAX_COUNT = 500;
	double qlevel = 0.01;
	double minDist = 10;
	bool needToInit = false;
	bool nightMode = false;

	


	VideoCapture cap1;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);


	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];







	namedWindow("LK Demo", 1);
	setMouseCallback("LK Demo", onMouse, 0);

	Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
	p0 = scene_points[0]; //objMatcher.center_points; //scene_points[0];
 	old_frame = scene.image.clone();
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
	Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

	cv::VideoCapture cap("video.mov");
	if (cap.isOpened())
	{
		for (;;)
		{
			Mat frame, frame_gray;
			cap >> frame;
			if (frame.empty())
				break;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	
			vector<uchar> status;
			vector<float> err;
			
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 20, 0.03);
			calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(31, 31), 3, criteria ,0, 0.001);
			
			vector<Point2f> good_new;
			for (uint i = 0; i < p0.size(); i++)
			{
				// Select good points
				if (status[i] == 1) {
					good_new.push_back(p1[i]);
					// draw the tracks
					line(mask, p1[i], p0[i], objects[0].color, 2);
					circle(frame, p1[i], 5, objects[0].color, -1);
				}
			}
			Mat img;
			add(frame, mask, img);
			imshow("Frame", img);
			int keyboard = waitKey(1);
			if (keyboard == 'q' || keyboard == 27)
				break;
			// Now update the previous frame and previous points
			old_gray = frame_gray.clone();
			p0 = good_new;
		}






		



		/*
			if (needToInit)
			{
				// automatic initialization
				goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
				addRemovePt = false;
			}
			else if (!points[0].empty())
			{
				vector<uchar> status;
				vector<float> err;

				if (prevGray.empty()) gray.copyTo(prevGray);

				calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
					3, termcrit, 0, 0.001);
				size_t i, k;
				for (i = k = 0; i < points[1].size(); i++)
				{
					if (addRemovePt)
					{
						if (norm(point - points[1][i]) <= 5)
						{
							addRemovePt = false;
							continue;
						}
					}
					if (!status[i]) continue;
					points[1][k++] = points[1][i];
					circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
				}
				points[1].resize(k);
			}

			if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
			{
				vector<Point2f> tmp;
				tmp.push_back(point);
				cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
				points[1].push_back(tmp[0]);
				addRemovePt = false;
			}
			needToInit = false;
			imshow("LK Demo", image);
			char c = (char)waitKey(10);
			if (c == 27) break;
			switch (c)
			{
			case 'r':
				needToInit = true;
				break;
			case 'c':
				points[0].clear();
				points[1].clear();
				break;
			case 'n':
				nightMode = !nightMode;
				break;
			}
			std::swap(points[1], points[0]);
			cv::swap(prevGray, gray);
		}
		*/
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