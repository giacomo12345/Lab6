/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */

#include <iostream>
#include "object_recognition.h"

using namespace cv;
using namespace std;

#define LOAD_VIDEO		0		// load all video frames or only the first
#define SHOW_FEATURES	0		// show keypoints or not


//---------------------------------------------------------------------------functions

std::vector<cv::Mat> loadVideo(String videoName, int toogleLoading);
vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio);


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
			String index = to_string(i);
			obj.showKeypoints("index" + index);
		}
	}

	/* create the object myMatcher with the vector of object to be recognized and the scene object */
	myMatcher objMatcher(objects, scene);

	/* compute the matches */
	objMatcher.computeMatches();

	////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// CLASSI FINO A QUI /////////////////////////////////////
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//

	/* select the matches according to the distance */
	std::vector< std::vector<DMatch> > matches = objMatcher.matches;
	std::vector<std::vector<DMatch>> good_matches;

	for (int i = 0; i < objects.size(); i++) {

		float ratio = 1.5;
		std::vector<DMatch> temp_good_matches;

		/* Find the minumun distance between matchpoints */
		float dist;
		dist = matches[i][0].distance;
		for (int j = 0; j < matches[i].size(); j++) {
			if (matches[i][j].distance < dist) dist = matches[i][j].distance;
		}

		/* Adapt the ratio in order to get at least 100 matches per object */
		float temp_ratio = ratio;
		while (temp_good_matches.size() < 100) {
			temp_good_matches = autotune_matches(matches[i], dist, temp_ratio);
			temp_ratio = 1.5 * temp_ratio;
			if (temp_good_matches.size() == matches[i].size()) break;
		}
		good_matches.push_back(temp_good_matches);
	}

	
	/* vettore di image matches - disegna le righe dei 4 oggetti rispetto alla scena*/
	std::vector<cv::Mat> img_matches;
	for (int i = 0; i < objects.size(); i++) {
		Mat temp_img_matches;
		drawMatches(objects[i].image, objects[i].getKeypoints(), scene.image, scene.getKeypoints(), good_matches[i], temp_img_matches, objects[i].color,
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		img_matches.push_back(temp_img_matches);
	}

	/* PER VEDERE IL RISULTATO, SELEZIONARE IL NUEMRO DELL'OGGETTO CON num_obj */
	//int num_obj = 0;
	//resize(img_matches[num_obj], img_matches[num_obj], Size(img_matches[num_obj].cols / 2, img_matches[num_obj].rows / 2));
	//imshow("img_matches", img_matches[num_obj]);

	/* vectors for store the results */
	std::vector<std::vector<Point2f>> obj_points;
	std::vector<std::vector<Point2f>> scene_points;
	std::vector<std::vector<Point2f>> obj_corners;
	std::vector<std::vector<Point2f>> scene_corners;
	std::vector<Mat> H;

	std::vector< std::vector<float>> coord;// X min & max , Y min & max

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			coord[i].push_back(0.0f);
		}
		coord[i][0] = objects[i].getKeypoints()[0].pt.x;
		coord[i][2] = objects[i].getKeypoints()[0].pt.y;
	}
	for (int i = 0; i < objects[i].getKeypoints().size(); i++)
	{

		float x = objects[i].getKeypoints()[i].pt.x;
		float y = objects[i].getKeypoints()[i].pt.y;

		if (x < coord[i][0]) coord[i][0] = x;
		else if (x > coord[i][1]) coord[i][1] = x;
		if (y < coord[i][2]) coord[i][2] = y;
		else if (y > coord[i][3]) coord[i][3] = y;

		cout << "keypoints x coordinate:     " << x << std::endl;
		cout << "keypoints y coordinate:     " << y << std::endl;
	}

	/* localize object - compute homography - get the corners from the object to be detected */
	for (int i = 0; i < objects.size(); i++) {

		/* temp elementes - will be stored in the outside vectors */
		std::vector<Point2f> temp_obj_points;
		std::vector<Point2f> temp_scene_points;
		std::vector<Point2f> temp_obj_corners(4);
		std::vector<Point2f> temp_scene_corners(4);

		/* Localize the object */
		for (size_t j = 0; j < good_matches[i].size(); j++) {
			temp_obj_points.push_back(objects[i].getKeypoints()[good_matches[i][j].queryIdx].pt);
			temp_scene_points.push_back(scene.getKeypoints()[good_matches[i][j].trainIdx].pt);
		}

		/* compute homography */
		Mat temp_H = findHomography(temp_obj_points, temp_scene_points, RANSAC);

		/* Get the corners from the object to be detected */
		temp_obj_corners[0] = Point2f(coord[i][0], coord[i][2]);
		temp_obj_corners[1] = Point2f(coord[i][0], coord[i][3]);
		temp_obj_corners[2] = Point2f(coord[i][1], coord[i][3]);
		temp_obj_corners[3] = Point2f(coord[i][1], coord[i][2]);

		perspectiveTransform(temp_obj_corners, temp_scene_corners, temp_H);

		/* store the "temp" elements in the outside vectors */
		obj_points.push_back(temp_obj_corners);
		scene_points.push_back(temp_scene_corners);
		obj_corners.push_back(temp_obj_corners);
		scene_corners.push_back(temp_scene_corners);
		H.push_back(temp_H);
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
/* ------------------------------------------------------------------------------------------*/
vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio) {
	vector<DMatch> match;
	for (int j = 0; j < int(Matches.size()); j++) {
		if (Matches[j].distance < min_dist * ratio) match.push_back(Matches[j]);
	}
	return match;
}
/* ------------------------------------------------------------------------------------------*/

