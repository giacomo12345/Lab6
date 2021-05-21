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


/* auto-resize the ratio for the filter */
vector<DMatch> autotunes_matches(vector<DMatch> Matches, float min_dist, float ratio) {
	vector<DMatch> match;
	for (int j = 0; j < int(Matches.size()); j++) {
		if (Matches[j].distance < min_dist * ratio) match.push_back(Matches[j]);
	}
	return match;
}


/* select the matches according to the distance - ratio and min nnumber of features founded*/
void myMatcher::filterMatches(float ratio, int features_min) {
	for (int i = 0; i < obj.size(); i++) {
		std::vector<DMatch> temp_good_matches;

		/* Find the minumun distance between matchpoints */
		float dist;
		dist = matches[i][0].distance;
		for (int j = 0; j < matches[i].size(); j++) {
			if (matches[i][j].distance < dist) dist = matches[i][j].distance;
		}

		/* Adapt the ratio in order to get at least features_min matches per object */
		float temp_ratio = ratio;
		while (temp_good_matches.size() < features_min) {
			temp_good_matches = autotunes_matches(matches[i], dist, temp_ratio);
			temp_ratio = 2 * temp_ratio;
			if (temp_good_matches.size() == matches[i].size()) break;
		}
		good_matches.push_back(temp_good_matches);
	}
}

/* return a vector of images where each keypoints of object are connected to the matched scene keypoints */
std::vector<cv::Mat>  myMatcher::getImageMatched() {
	std::vector<cv::Mat> img_matches;
	for (int i = 0; i < obj.size(); i++) {
		Mat temp_img_matches;
		drawMatches(obj[i].image, obj[i].getKeypoints(), scene.image, scene.getKeypoints(), good_matches[i], temp_img_matches, obj[i].color,
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		img_matches.push_back(temp_img_matches);
	}
	return img_matches;
}

/* localize object and  compute homography  */
void myMatcher::computeHomography() {
	for (int i = 0; i < obj.size(); i++) {

		/* temp elementes - will be stored in the outside vectors */
		std::vector<Point2f> temp_obj_points;
		std::vector<Point2f> temp_scene_points;

		/* Localize the object */
		for (size_t j = 0; j < good_matches[i].size(); j++) {
			temp_obj_points.push_back(obj[i].getKeypoints()[good_matches[i][j].queryIdx].pt);
			temp_scene_points.push_back(scene.getKeypoints()[good_matches[i][j].trainIdx].pt);
		}

		/* compute homography */
		Mat temp_H = findHomography(temp_obj_points, temp_scene_points, RANSAC);

		/* store the "temp" elements in the outside vectors */
		obj_points.push_back(temp_obj_points);
		scene_points.push_back(temp_scene_points);
		H.push_back(temp_H);
	}
}

/* find the object images corners to draw the rettangle */
void myMatcher::findCorners() {

	std::vector< std::vector<float>> coord;
	for (int i = 0; i < obj.size(); i++) {
		std::vector<float > temp_coord;
		temp_coord.push_back(obj[i].getKeypoints()[0].pt.x);
		temp_coord.push_back(obj[i].getKeypoints()[0].pt.x);
		temp_coord.push_back(obj[i].getKeypoints()[0].pt.y);
		temp_coord.push_back(obj[i].getKeypoints()[0].pt.y);
		coord.push_back(temp_coord);
	}

	for (int j = 0; j < obj.size(); j++) {
		for (int i = 0; i < obj[j].getKeypoints().size(); i++) {

			float x = obj[j].getKeypoints()[i].pt.x;
			float y = obj[j].getKeypoints()[i].pt.y;

			if (x < coord[j][0]) coord[j][0] = x;
			else if (x > coord[j][1]) coord[j][1] = x;

			if (y < coord[j][2]) coord[j][2] = y;
			else if (y > coord[j][3]) coord[j][3] = y;
		}
	}

	for (int i = 0; i < obj.size(); i++) {
		std::vector<Point2f> temp_obj_corners(4);
		temp_obj_corners[0] = Point2f(0, 0);
		temp_obj_corners[1] = Point2f((float)obj[i].image.cols, 0);
		temp_obj_corners[2] = Point2f((float)obj[i].image.cols, (float)obj[i].image.rows);
		temp_obj_corners[3] = Point2f(0, (float)obj[i].image.rows);

		temp_obj_corners[0] = Point2f(coord[i][0], coord[i][2]);
		temp_obj_corners[1] = Point2f(coord[i][0], coord[i][3]);
		temp_obj_corners[2] = Point2f(coord[i][1], coord[i][3]);
		temp_obj_corners[3] = Point2f(coord[i][1], coord[i][2]);

		temp_obj_corners[0] = Point2f(0, 0);
		temp_obj_corners[1] = Point2f((float)obj[i].image.cols, 0);
		temp_obj_corners[2] = Point2f((float)obj[i].image.cols, (float)obj[i].image.rows);
		temp_obj_corners[3] = Point2f(0, (float)obj[i].image.rows);

		obj_corners.push_back(temp_obj_corners);
	}
}

/* compute the projection from the object images to the scene image */
void myMatcher::computeProjection() {
	for (int i = 0; i < obj.size(); i++) {
		std::vector<Point2f> temp_scene_corners(4);
		perspectiveTransform(obj_corners[i], temp_scene_corners, H[i]);
		scene_corners.push_back(temp_scene_corners);
	}
}

/* return the scene points */
std::vector < std::vector<Point2f> > myMatcher::getScenePoints() {
	return scene_points;
}

/* return the corners points */
std::vector<std::vector<Point2f>> myMatcher::getSceneCorners() {
	return scene_corners;
}

/* compute the center point of the object in the scene and the max distance from this point and other points of the image*/
void  myMatcher::computeCenterPoints() {

	for(int i = 0 ; i < obj.size() ; i++) {

		/* vectors for store center point and corner point */
		std::vector<Point2f> points_obj;
		std::vector<Point2f> points_scene;

		Point2f temp_point_obj;
		Point2f temp_point_scene;
		temp_point_obj.y = obj[i].image.rows/2;
		temp_point_obj.x = obj[i].image.cols/2;
		points_obj.push_back(temp_point_obj);
		points_obj.push_back(Point2f(0,0));

		perspectiveTransform(points_obj, points_scene, H[i]);

		Point2f diag = points_scene[0] - points_scene[1];
		float center_dist = sqrt(diag.y*diag.y + diag.x* diag.x);
	
		max_distance.push_back(center_dist*1.1);
		center_points.push_back(points_scene[0]);
	}
}