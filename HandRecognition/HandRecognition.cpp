#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void fixIntensity(Mat src, Mat &dst) {
	Mat hsv, hsv_final, temp;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	vector<Mat> split_hsv;
	split(hsv, split_hsv);

	split_hsv[2] = 255;

	merge(split_hsv, hsv_final);

	cvtColor(hsv_final, dst, COLOR_HSV2BGR);
}

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

void convertBinary(Mat src, Mat &dst) {
	Mat gray;
	cvtColor(src, gray, COLOR_RGB2GRAY);
	threshold(gray, dst, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
}

void getContoursAndRoi(Mat binary, vector<vector<Point>> &contours, vector<Vec4i> hierarchy, Rect &roi) {
	int min_x = 99999, min_y = 99999, max_x = 0, max_y = 0;
	int max_contour = 0;
	int max_contour_index = NULL;

	findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 1) {
		for (size_t i = 1; i < contours.size(); i++) {
			contours[0].insert(contours[0].end(), contours[i].begin(), contours[i].end());
		}
	}

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
		mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float) (mu[i].m01 / mu[i].m00));
	}

	centroid = mc[0];
}

double calcDistance(Point p1, Point p2) {
	double x = abs(p1.x - p2.x);
	double y = abs(p1.y - p2.y);

	return sqrt(pow(x, 2) + pow(y, 2));
}

vector<Point> getValidPoints(vector<vector<Point>> contours, Point2f centroid, String orientation) {
	vector<Point> valid_points, res;

	if (orientation == "vertical") {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (contours[0][i].y <= centroid.y)
				valid_points.push_back(contours[0][i]);
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
			if (contours[0][i].x >= centroid.x)
				valid_points.push_back(contours[0][i]);
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

double calcMedianMinDistance(vector<vector<Point>> contours, Point2f centroid, String orientation) {
	vector<Point> valid_points, valeys;

	double min_distance = 99999;

	if (orientation == "vertical") {
		for (size_t i = 0; i < contours[0].size(); i++) {
			if (contours[0][i].y <= centroid.y)
				valid_points.push_back(contours[0][i]);
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
			if (contours[0][i].x >= centroid.x)
				valid_points.push_back(contours[0][i]);
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

void getFingerTips(Mat image, vector<vector<Point>> contours, Point2f centroid, vector<Point> &finger_tips, String orientation) {
	vector<Point> valid_points = getValidPoints(contours, centroid, orientation);

	vector<vector<Point>> repeated = divideByRegion(image, valid_points, orientation);

	double medianMinDistance = calcMedianMinDistance(contours, centroid, orientation);

	double minFingerTipSize;

	if (orientation == "vertical")
		minFingerTipSize = image.rows / 4;
	else
		minFingerTipSize = image.cols / 4;

	finger_tips = getSinglePoints(repeated, centroid, medianMinDistance, minFingerTipSize);
	//finger_tips = closest_points;
}

void drawFingerTipsCircles(Mat &img, vector<Point> finger_tips) {
	for (int i = 0; i < finger_tips.size(); i++) {
		circle(img, (Point)finger_tips[i], 4, Scalar(0, 0, 255), 2);
	}
}

/*
	Creates a box around the contour by applying minAreaRect.
	source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rotated_ellipses/bounding_rotated_ellipses.html
*/
vector<RotatedRect> getsRectBoxes(vector<vector<Point>> contours) {
	vector<RotatedRect> minRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
	}
	return minRect;
}

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

/**  @function main */
int main(int argc, char** argv)
{
	vector<Point> finger_tips;
	vector<vector<Point>> contours, cropped_contours;
	vector<Vec4i> hierarchy;
	Point2f centroid;
	Mat src, fixed_intensity, noise_removed, binary, cropped_image, cropped_img_rgb;
	Rect roi;
	string orientation;

	/// Load image
	src = imread("C:\\Users\\Joao\\source\\repos\\openCvTry\\openCvTry\\hand8.png");

	fixIntensity(src, fixed_intensity);

	removeNoise(fixed_intensity, noise_removed);

	convertBinary(noise_removed, binary);

	getContoursAndRoi(binary, contours, hierarchy, roi);

	cropped_image = binary(roi);

	calcCentroid(cropped_image, cropped_contours, hierarchy, centroid);

	cvtColor(cropped_image, cropped_img_rgb, COLOR_GRAY2BGR);

	circle(cropped_img_rgb, (Point) centroid, 4, Scalar(127, 127, 127), 2);

	orientation = getOrientation(cropped_contours);

	getFingerTips(cropped_img_rgb, cropped_contours, centroid, finger_tips, orientation);

	drawFingerTipsCircles(cropped_img_rgb, finger_tips);

	imshow("centroid", cropped_img_rgb);
	/// Wait until user exits the program*/
	waitKey(0);

	return 0;
}