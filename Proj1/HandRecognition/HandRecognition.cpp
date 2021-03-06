#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Fix image intensity
/**
 * @param src
 *    Source image
 * @param dst
 *	  Destination image with fixed intensity
 */
void fixIntensity(Mat src, Mat &dst) {
	Mat hsv, hsv_final, temp;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	vector<Mat> split_hsv;
	split(hsv, split_hsv);

	split_hsv[2] = 255;

	merge(split_hsv, hsv_final);

	cvtColor(hsv_final, dst, COLOR_HSV2BGR);
}

// Remove image noise by using Gaussian Blur
/**
 * @param src
 *    Source image
 * @param dst
 *	  Destination image without noise
 */
void removeNoise(Mat src, Mat &dst) {
	for (int i = 1; i < 15; i = i + 2)
	{
		GaussianBlur(src, src, Size(i, i), 0, 0);
	}

	Mat ycrcb, ycrcb_final;

	cvtColor(src, ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> spl;
	split(ycrcb, spl);

	spl[1] = 0;
	spl[2] = 0;

	merge(spl, ycrcb_final);
	cvtColor(ycrcb_final, dst, COLOR_YCrCb2BGR);
}

// Converts an image to binary
/**
 * @param src
 *    Source image in RGB color space
 * @param dst
 *	  Destination binary image
 */
void convertBinary(Mat src, Mat &dst) {
	Mat gray;
	cvtColor(src, gray, COLOR_RGB2GRAY);
	threshold(gray, dst, 127, 255, THRESH_BINARY_INV | THRESH_OTSU);
}

// Gets image contours and region of interest, in this case the hand
/**
 * @param binary
 *	  Binary image
 * @param contours
 *	  Image contours
 * @param hierarchy 
 *    Optional output vector, containing information about the image topology 
 * @param roi
 *	  Region of interest of the image
 */
void getContoursAndRoi(Mat binary, vector<vector<Point>> &contours, vector<Vec4i> hierarchy, Rect &roi) {
	int min_x = 99999, min_y = 99999, max_x = 0, max_y = 0;
	int max_contour = 0;
	int max_contour_index = NULL;

	findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> contour;
	int max_size = 0, index = 0;

	for (size_t i = 1; i < contours.size(); i++) {
			contours[0].insert(contours[0].end(), contours[i].begin(), contours[i].end());
			if (contours[i].size() > max_size) {
				max_size = contours[i].size();
				index = i;
			}
	}

	contour.push_back(contours[index]);

	contours = contour;

	for (size_t j = 0; j < contours[0].size(); j++) {
		Point p = contours[0][j];
		if (p.x < min_x)
			min_x = p.x;
		if (p.x > max_x)
			max_x = p.x;
		if (p.y < min_y)
			min_y = p.y;
		if (p.y > max_y)
			max_y = p.y;
	}

	roi = Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

// Calculates the image centroid
/**
 * @param cropped_image
 *    Cropped image
 * @param cropped_contours
 *	  Contours of the cropped image
 * @param hierarchy
 *    Optional output vector, containing information about the image topology
 * @param centroid
 *    Coordinates of the centroid
 */
void calcCentroid(Mat cropped_image, vector<vector<Point>> &cropped_contours, vector<Vec4i> hierarchy, Point2f &centroid) {
	findContours(cropped_image, cropped_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<Moments> mu(cropped_contours.size());
	for (int i = 0; i < cropped_contours.size(); i++)
	{
		mu[i] = moments(cropped_contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(cropped_contours.size());
	for (int i = 0; i < cropped_contours.size(); i++)
	{
		mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float)(mu[i].m01 / mu[i].m00));
	}

	centroid = mc[0];
}

// Calculates euclidean distance between two points
/**
 * @param p1
 *    First point
 * @param p2
 *	  Second point
 */
double calcDistance(Point p1, Point p2) {
	double x = abs(p1.x - p2.x);
	double y = abs(p1.y - p2.y);

	return sqrt(pow(x, 2) + pow(y, 2));
}

// Gets the image valid points, according to the centroid position, hand orientation and hand origin. These points
// are local maximums.
/**
 * @param contours
 *    Image contours
 * @param centroid
 *	  Coordinates of the centroid
 * @param orientation
 *    Image orientation (vertical or horizontal)
 * @param origin
 *    Image origin (from where to where is the hand going i.e. right to left)
 */
vector<Point> getValidPoints(vector<vector<Point>> contours, Point2f centroid, String orientation, String origin) {
	vector<Point> valid_points, res;

	if (orientation == "vertical") {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (origin == "d2u") {
				if (contours[0][i].y <= centroid.y)
					valid_points.push_back(contours[0][i]);
			}
			else if (origin == "u2d") {
				if (contours[0][i].y >= centroid.y)
					valid_points.push_back(contours[0][i]);
			}
		}

		bool IS_INCREASING = true;

		for (size_t j = 1; j < valid_points.size() - 1; j++) {
			double distance = calcDistance(valid_points[j], centroid);
			double next_distance = calcDistance(valid_points[j + 1], centroid);
			double previous_distance = calcDistance(valid_points[j - 1], centroid);

			if (next_distance < distance && distance > previous_distance) {
				res.push_back(valid_points[j]);
				IS_INCREASING = false;
			}
		}
	}
	else {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (origin == "l2r") {
				if (contours[0][i].x >= centroid.x)
					valid_points.push_back(contours[0][i]);
			}
			else if (origin == "r2l") {
				if (contours[0][i].x <= centroid.x)
					valid_points.push_back(contours[0][i]);
			}
		}

		bool IS_INCREASING = true;

		for (size_t j = 1; j < valid_points.size() - 1; j++) {
			double distance = calcDistance(valid_points[j], centroid);
			double next_distance = calcDistance(valid_points[j + 1], centroid);
			double previous_distance = calcDistance(valid_points[j - 1], centroid);

			if (next_distance < distance && distance > previous_distance) {
				res.push_back(valid_points[j]);
				IS_INCREASING = false;
			}
		}
	}

	return res;
}

// Divides the image in 7 parts and groups the points which distance is less than the region size
/**
 * @param image
 *    Image to divide
 * @param valid_points
 *	  Image valid points
 * @param orientation
 *    Image orientation (vertical or horizontal)
 * @return
 *    Vector with points separated by region
 */
vector<vector<Point>> divideByRegion(Mat image, vector<Point> valid_points, String orientation) {
	int delta;

	if (orientation == "vertical")
		delta = image.cols / 7;
	else
		delta = image.rows / 7;

	vector<vector<Point>> repeated;

	bool IS_EQUIVALENT = true;
	bool MATCH = false;

	for (size_t i = 0; i < valid_points.size(); i++) {
		Point p = valid_points[i];
		if (repeated.size() == 0) {
			vector<Point> v;
			repeated.push_back(v);
			repeated[repeated.size() - 1].push_back(p);
		}
		else {
			for (size_t k = 0; k < repeated.size(); k++) {
				IS_EQUIVALENT = true;
				for (size_t j = 0; j < repeated[k].size(); j++) {
					Point q = repeated[k][j];
					if (orientation == "vertical") {
						if (abs(p.x - q.x) > delta) {
							IS_EQUIVALENT = false;
							break;
						}
					}
					else {
						if (abs(p.y - q.y) > delta) {
							IS_EQUIVALENT = false;
							break;
						}
					}
				}

				if (IS_EQUIVALENT) {
					repeated[k].push_back(p);
					MATCH = false;
					break;
				}
			}

			if (!MATCH && !IS_EQUIVALENT) {
				vector<Point> v;
				repeated.push_back(v);
				IS_EQUIVALENT = true;
				repeated[repeated.size() - 1].push_back(p);
			}
		}
	}

	return repeated;
}

// Analyzes the groups of points and calculates the points that are finger tips
/**
 * @param repeated
 *    Vector of vectors of points with the groups of points
 * @param centroid
 *	  Coordinates of the centroid
 * @param min_distance
 *    Minimum distance from the image valeys to the centroid
 * @param min_finger_tip_distance
 *    Minimum size of a finger
 * @return
 *    Vector with points that are the finger tips
 */
vector<Point> getSinglePoints(vector<vector<Point>> repeated, Point2f centroid, double min_distance, double min_finger_tip_distance) {
	vector<Point> single_points, res;

	double max_overall_distance = 0.0;

	for (size_t k = 0; k < repeated.size(); k++) {
		int max_index = 0;
		double max_distance = 0.0;
		for (size_t l = 0; l < repeated[k].size(); l++) {
			Point p = repeated[k][l];
			double distance = calcDistance(p, centroid);
			if (distance > max_distance) {
				max_index = (int)l;
				max_distance = distance;

				if (max_distance > max_overall_distance)
					max_overall_distance = max_distance;
			}
		}
		single_points.push_back(repeated[k][max_index]);
	}

	for (size_t m = 0; m < single_points.size(); m++) {
		Point p = single_points[m];
		double distance = calcDistance(p, centroid);

		if (distance >= 0.75 * max_overall_distance && distance - min_distance > min_finger_tip_distance)
			res.push_back(p);
	}

	return res;
}

// Calculates the median minimum distance from the hand valeys to the centroid
/**
 * @param contours
 *    Contours of the image
 * @param centroid
 *	  Coordinates of the centroid
* @param orientation
 *    Image orientation (vertical or horizontal)
  * @param origin
 *    Image origin (from where to where is the hand going i.e. right to left)
 * @return
 *    Medium minimum distance
 */
double calcMedianMinDistance(vector<vector<Point>> contours, Point2f centroid, String orientation, String origin) {
	vector<Point> valid_points, valeys;

	double min_distance = 99999;

	if (orientation == "vertical") {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (origin == "d2u") {
				if (contours[0][i].y <= centroid.y)
					valid_points.push_back(contours[0][i]);
			}
			else if (origin == "u2d") {
				if (contours[0][i].y >= centroid.y)
					valid_points.push_back(contours[0][i]);
			}
		}

		bool IS_INCREASING = true;

		for (size_t j = 1; j < valid_points.size() - 1; j++) {
			double distance = calcDistance(valid_points[j], centroid);
			double next_distance = calcDistance(valid_points[j + 1], centroid);
			double previous_distance = calcDistance(valid_points[j - 1], centroid);

			if (next_distance > distance && distance < previous_distance) {
				if (distance < min_distance) {
					min_distance = distance;
				}
				valeys.push_back(valid_points[j]);
				IS_INCREASING = false;
			}
		}
	}
	else {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (origin == "l2r") {
				if (contours[0][i].x >= centroid.x)
					valid_points.push_back(contours[0][i]);
			}
			else if (origin == "r2l") {
				if (contours[0][i].x <= centroid.x)
					valid_points.push_back(contours[0][i]);
			}
		}

		bool IS_INCREASING = true;

		for (size_t j = 1; j < valid_points.size() - 1; j++) {
			double distance = calcDistance(valid_points[j], centroid);
			double next_distance = calcDistance(valid_points[j + 1], centroid);
			double previous_distance = calcDistance(valid_points[j - 1], centroid);

			if (next_distance > distance && distance < previous_distance) {
				if (distance < min_distance) {
					min_distance = distance;
				}
				valeys.push_back(valid_points[j]);
				IS_INCREASING = false;
			}
		}
	}

	int distances_size = 0;
	double distances_sum = 0;

	for (size_t i = 0; i < valeys.size(); i++) {
		double distance = calcDistance(valeys[i], centroid);

		if (distance < 1.3 * min_distance) {
			distances_sum += distance;
			distances_size++;
		}
	}

	return distances_sum / distances_size;
}

// Finds the finger tips of the image
/**
 * @param image
 *    Image analyzed
 * @param contours
 *    Contours of the image
 * @param centroid
 *	  Coordinates of the centroid
 * @param finger_tips
 *	Vector with the finger tips points
* @param orientation
 *    Image orientation (vertical or horizontal)
  * @param origin
 *    Image origin (from where to where is the hand going i.e. right to left)
 */
void getFingerTips(Mat image, vector<vector<Point>> contours, Point2f centroid, vector<Point> &finger_tips, String orientation, String origin) {
	vector<Point> valid_points = getValidPoints(contours, centroid, orientation, origin);

	vector<vector<Point>> repeated = divideByRegion(image, valid_points, orientation);

	double medianMinDistance = calcMedianMinDistance(contours, centroid, orientation, origin);

	double minFingerTipSize;

	if (orientation == "vertical")
		minFingerTipSize = image.rows / 5;
	else
		minFingerTipSize = image.cols / 5;

	finger_tips = getSinglePoints(repeated, centroid, medianMinDistance, minFingerTipSize);
}

// Draws circles in each fingertip
/**
 * @param image
 *    Image to draw
 * @param finger_tips
 *	  Vector with finger tip points
 */
void drawFingerTipsCircles(Mat &img, vector<Point> finger_tips) {
	for (int i = 0; i < finger_tips.size(); i++) {
		circle(img, (Point)finger_tips[i], 4, Scalar(0, 0, 255), 2);
	}
}

/* Creates a box around the contour by applying minAreaRect.
 source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rotated_ellipses/bounding_rotated_ellipses.html
*/
// Gets a box around the hand
/**
 * @param contours
 *    Image contours
 */
vector<RotatedRect> getsRectBoxes(vector<vector<Point>> contours) {
	vector<RotatedRect> minRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
	}
	return minRect;
}

// Gets image orientation
/**
 * @param contours
 *    Image contours
 * @return
 *    Image orientation (vertical or horizontal)
 */
string getOrientation(vector<vector<Point>> contours) {
	vector<RotatedRect> minRect = getsRectBoxes(contours);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Rect temp = minRect[i].boundingRect();
		if (temp.height > temp.width) {
			return "vertical";
		}
		else {
			return "horizontal";
		}
	}
}

// Displays the final result
/**
 * @param numberOfFingers
 *    Number of fingers
 * @param orientation
 *    Hand orientation
 */
int displayResult(string numberOfFingers, string orientation)
{
	Size textsize = getTextSize(numberOfFingers, CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
	Size textsize1 = getTextSize(orientation, CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
	Mat image = Mat::zeros(300, textsize1.width, CV_8UC3);
	Point org((image.size().width - textsize.width) / 2, (300 - textsize.height) / 2 + 10);
	Point org2(5, (300 - textsize.height) / 2 + textsize.height);
	int lineType = 8;

	Mat image2;

	for (int i = 0; i < 255; i += 2)
	{
		image2 = image - Scalar::all(i);
		putText(image2, numberOfFingers, org, CV_FONT_HERSHEY_COMPLEX, 3,
			Scalar(i, i, 255), 5, lineType);
		putText(image2, orientation, org2, CV_FONT_HERSHEY_COMPLEX, 3,
			Scalar(i, i, 255), 5, lineType);

		imshow("Result", image2);
		if (waitKey(5) >= 0)
		{
			return -1;
		}
	}
	return 0;
}

// Checks if there are finger tips detected in a region
/**
 * @param finger_tips
 *    Vector with the finger tips points
 * @param img
 *    Region analyzed
 * @param side
 *	  Side of the image analyzed (can be 'left' or 'right')
 */
bool checkFingerTipsInRegion(vector<Point> finger_tips, Mat img, String side) {

	int min_x, min_y, max_x, max_y;

	if (side == "left") {
		min_x = 0;
		min_y = 0;
		max_x = 30;
		max_y = img.rows;
	}
	else {
		min_x = img.cols - 30;
		min_y = 0;
		max_x = img.cols;
		max_y = img.rows;
	}

	for (size_t i = 0; i < finger_tips.size(); i++) {
		Point p = finger_tips[i];
		if (p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y)
			return true;
	}

	return false;
}

// Checks if there is a thumb detected in the image
/**
 * @param finger_tips
 *    Vector with the finger tips points
 * @param orientation
 *    Hand orientation
 */
int checkThumb(Mat img, vector<Point> finger_tips, String orientation) {

	Mat left, right;

	if (orientation == "vertical") {
		left = img(Rect(0, 0, 30, img.rows));
		right = img(Rect(img.cols - 30, 0, 30, img.rows));
	}
	else {
		left = img(Rect(0, 0, img.cols, 30));
		right = img(Rect(0, img.rows - 30, img.cols, 30));
	}
	
	int pixels = left.rows* left.cols;

	int white_left = countNonZero(left);
	int white_right = countNonZero(right);

	bool DETECTED_LEFT = false, DETECTED_RIGHT = false;

	if (white_left < 0.05 * pixels)
		DETECTED_LEFT = true;
	if (white_right < 0.05 * pixels)
		DETECTED_RIGHT = true;

	if (DETECTED_LEFT) {
		if (checkFingerTipsInRegion(finger_tips, img, "left"))
			DETECTED_LEFT = false;
	}

	if (DETECTED_RIGHT) {
		if (checkFingerTipsInRegion(finger_tips, img, "right"))
			DETECTED_RIGHT = false;
	}

	if (DETECTED_LEFT && DETECTED_RIGHT)
		return 1;
	else if (DETECTED_LEFT || DETECTED_RIGHT)
		return 1;

	return 0;
}

// Gets hand origin
/**
 * @param image
 *    Image with hand
 * @param orientation
 *    Image orientation (vertical or horizontal)
 * @return
 *    Hand origin (i.e. Right to left -> r2l)
 */
String getHandOrigin(Mat image, String orientation) {
	if (orientation == "horizontal") {
		if (countNonZero(image.col(0)) < countNonZero(image.col(image.cols - 1))) {
			return "r2l";
		}
		else {
			return "l2r";
		}
	}
	else {
		if (countNonZero(image.row(0)) < countNonZero(image.row(image.rows - 1))) {
			return "d2u";
		}
		else {
			return "u2d";
		}
	}
}


// Captures image from webcam
Mat captureImage() {
	VideoCapture cap;

	if (cap.open(0)) {
		int i = 0;
		while (true)
		{
			i++;
			Mat frame;
			cap >> frame;
			if (frame.empty()) break; // end of video stream
			imshow("capture", frame);

			if (waitKey(30) >= 0) {
				String name = to_string(i) + ".png";
				imwrite(name, frame);
				return frame;
			}
		}
	}
}

/**  @function main */
int main(int argc, char** argv)
{
	vector<Point> finger_tips;
	vector<vector<Point>> contours, cropped_contours;
	vector<Vec4i> hierarchy;
	Point2f centroid;
	Mat src, fixed_intensity, noise_removed, binary, cropped_image, cropped_img_rgb;
	Rect roi;
	string orientation, origin, imagePath;

	cout << "1- Type name of the file" << endl << endl << "2- Take picture with webcam" << endl << endl;

	int select;

	cin >> select;

	if (select == 1) {
		cout << "Write the name of the file: ";
		cin >> imagePath;
		src = imread(imagePath);
	}
	else {
		src = captureImage();
	}

	/// Load image

	fixIntensity(src, fixed_intensity);

	removeNoise(fixed_intensity, noise_removed);

	convertBinary(noise_removed, binary);

	getContoursAndRoi(binary, contours, hierarchy, roi);

	cropped_image = binary(roi);

	calcCentroid(cropped_image, cropped_contours, hierarchy, centroid);

	cvtColor(cropped_image, cropped_img_rgb, COLOR_GRAY2BGR);

	circle(cropped_img_rgb, (Point)centroid, 4, Scalar(127, 127, 127), 2);

	orientation = getOrientation(cropped_contours);

	origin = getHandOrigin(cropped_image, orientation);

	getFingerTips(cropped_img_rgb, cropped_contours, centroid, finger_tips, orientation, origin);

	int thumb = checkThumb(cropped_image, finger_tips, orientation);

	drawFingerTipsCircles(cropped_img_rgb, finger_tips);

	imshow("centroid", cropped_img_rgb);

	displayResult(to_string(finger_tips.size() + thumb), orientation);
	/// Wait until user exits the program*/
	waitKey(0);

	return 0;
}