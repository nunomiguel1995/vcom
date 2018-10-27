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

void getFingerTips(Mat image, vector<vector<Point>> contours, Point2f centroid, vector<Point> &finger_tips, String orientation) {
	vector<Point> valid_points, temp;
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
				temp.push_back(valid_points[j]);
				IS_INCREASING = false;
			}
		}
	}
	int delta = image.cols / 7;

	vector<vector<Point>> repeated;

	int index = 0;

	bool IS_EQUIVALENT = true;

	for (size_t i = 0; i < temp.size(); i++) {
		Point p = temp[i];
		if (repeated.size() == 0) {
			vector<Point> v;
			repeated.push_back(v);
			repeated[index].push_back(p);
		}
		else {
			for (size_t j = 0; j < repeated[index].size(); j++) {
				Point q = repeated[index][j];
				if (abs(p.x - q.x) > delta) {
					IS_EQUIVALENT = false;
					break;
				}
			}
			if (!IS_EQUIVALENT) {
				vector<Point> v;
				repeated.push_back(v);
				index++;
				IS_EQUIVALENT = true;
			}			
			repeated[index].push_back(p);
		}
	}

	vector<Point> single_points;

	double max_overall_distance = 0.0;

	for (size_t k = 0; k < repeated.size(); k++) {
		int max_index = 0;
		double max_distance = 0.0;
		for (size_t l = 0; l < repeated[k].size(); l++) {
			Point p = repeated[k][l];
			double distance = calcDistance(p, centroid);
			if (distance > max_distance) {
				max_index = (int) l;
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

		if (distance >= 0.75 * max_overall_distance)
			finger_tips.push_back(p);
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

	/// Load image
	src = imread("C:\\Users\\Joao\\source\\repos\\openCvTry\\openCvTry\\hand13.jpg");

	fixIntensity(src, fixed_intensity);

	removeNoise(fixed_intensity, noise_removed);

	convertBinary(noise_removed, binary);

	getContoursAndRoi(binary, contours, hierarchy, roi);

	cropped_image = binary(roi);

	calcCentroid(cropped_image, cropped_contours, hierarchy, centroid);

	cvtColor(cropped_image, cropped_img_rgb, COLOR_GRAY2BGR);

	circle(cropped_img_rgb, (Point) centroid, 4, Scalar(127, 127, 127), 2);

	//TODO: calculate orientation

	getFingerTips(cropped_img_rgb, cropped_contours, centroid, finger_tips, "vertical");
	
	int rows = cropped_img_rgb.cols;

	for (int i = 0; i < finger_tips.size(); i++) {
		circle(cropped_img_rgb, (Point)finger_tips[i], 4, Scalar(127, 127, 127), 2);
	}

	imshow("centroid", cropped_img_rgb);
	/// Wait until user exits the program*/
	waitKey(0);

	return 0;
}