/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO*/

#include <iostream>
#include "object_recognition.h"

using namespace cv;
using namespace std;

#define LOAD_VIDEO		0		// load all video frames or only the first
#define SHOW_FEATURES	0		// show keypoints or not


//-----------------------------------------------------functions
vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio) {
	vector<DMatch> match;
	for (int j = 0; j < int(Matches.size()); j++) {
		if (Matches[j].distance < min_dist * ratio) match.push_back(Matches[j]);
	}
	return match;
}

//------------------------------------------------------------------------------------
//                               MAIN FUNCTION
//------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {	// risolvere il problema di poter avere piu o meno di 4 immagini (opzionale)

	/* initializing vectors*/
	std::vector<cv::Mat> frames;
	std::vector<myObject> objects;
	std::vector < std::vector<cv::KeyPoint> > keypoints;	// poi non servirà
	std::vector<cv::Mat> descriptors;						// poi non servirà
	
	/* upload video frames */---------------------------------------------------------------------------------*/
	cout << "frames uploading";
	cv::VideoCapture cap("video.mov");
	if (cap.isOpened()) {
		int loader = 0;
		for (;;) {
			cv::Mat frame;
			cap >> frame;
			if (!cap.read(frame)) break;
			frames.push_back(frame);
			if (loader % 110 == 0) cout << "."; 
			loader++;
			if (!LOAD_VIDEO) break;
			/* namedWindow("video", WINDOW_FREERATIO);  imshow("video", frame); waitKey(1); */
		}
	}
	else std::cout << "error 404 video not found" << std::endl;
	cout << "number frames found: " << frames.size() << std::endl;
	cap.release();  //destroyWindow("video");
	/*--------------------------------------------------------------------------------------------------------*/
	
	
	
	/* create myObject with the first frame of the video*/
	myObject scene(frames[0]);
	
	
	/* upload images and create the vector of objects */
	for (int i = 1; i < 5; i++) {
		Mat img = imread(argv[i], IMREAD_COLOR);
		myObject obj(img);
		objects.push_back(obj);
		
		String index = to_string(i);
		keypoints.push_back(obj.getKeypoints());				// poi non servirà
		descriptors.push_back(obj.getDescriptors());				// poi non servirà
		if (SHOW_FEATURES) obj.showKeypoints("index" + index);
	}

	
	/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

	/* PER FAR FUNZIONARE IL RESTO SENZA LA CLASSE PER FARE IL MATCH CHE è DA FARE */
	std::vector<cv::Mat> src;
	for (int i = 0; i < 4; i++) src.push_back(objects[i].image);
	src.push_back(frames[0]);
	keypoints.push_back(scene.getKeypoints());
	descriptors.push_back(scene.getDescriptors());
	//----------------------------------------------------------------------------------



	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
	std::vector< std::vector<DMatch> > knn_matches;

	myMatcher(objects, scene);

	for (int i = 0; i < 4; i++)
	{
		vector<DMatch> tmp_matches;
		matcher->match(descriptors[i], descriptors[4], tmp_matches, Mat());
		knn_matches.push_back(tmp_matches);
	}





	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	keypoints_object = keypoints[0];
	keypoints_scene = keypoints[4];
	Mat img_object = src[0].clone();
	Mat img_scene = src[4].clone();;


	std::vector<DMatch> good_matches;
	/* COPIATO DA SITO DI GIACOMO */

	/*
	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.75f;

	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}*/
	float ratio = 1.5;

	float instance_ratio = ratio;

	// push in bestMatches the top 50 matched features
	vector<DMatch> bestMatches;
	int nbMatch = int(knn_matches[0].size());
	Mat tab(nbMatch, 1, CV_32F);

	float dist;
	float min_dist = -1.;

	// Find the minumun distance between matchpoints
	for (int j = 0; j < nbMatch; j++)
	{
		dist = knn_matches[0][j].distance;

		// update the minumun distance
		if (min_dist < 0 || dist < min_dist)
			min_dist = dist;
	}


	// Adapt the ratio in order to get at least 120 matches per couple of adjacent images
	do
	{
		good_matches = autotune_matches(knn_matches[0], min_dist, instance_ratio);
		instance_ratio = 2 * instance_ratio;
	} while (good_matches.size() < 120);



	//cout << "Size of " << i << "-th" << "TOP matches: " << BestMatches[i].size() << endl;




	//-- Draw matches
	Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, objects[0].color,
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene_points;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene_points.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}
	cout << "good_matches size:     " << good_matches.size() << std::endl;
	cout << "keypoints_object size:     " << keypoints_object.size() << std::endl;
	cout << "obj size:     " << obj.size() << std::endl;
	Mat H = findHomography(obj, scene_points, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_object.cols, 0);
	obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
	obj_corners[3] = Point2f(0, (float)img_object.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
		scene_corners[1] + Point2f((float)img_object.cols, 0), objects[0].color, 4);
	line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
		scene_corners[2] + Point2f((float)img_object.cols, 0), objects[0].color, 4);
	line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
		scene_corners[3] + Point2f((float)img_object.cols, 0), objects[0].color, 4);
	line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
		scene_corners[0] + Point2f((float)img_object.cols, 0), objects[0].color, 4);
	//-- Show detected matches
	namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
	resize(img_matches, img_matches, Size(img_matches.cols / 2, img_matches.rows / 2));
	imshow("Good Matches & Object detection", img_matches);


	cout << "END" << std::endl;



	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------
