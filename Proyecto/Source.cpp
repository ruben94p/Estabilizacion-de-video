#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

#include <chrono>

using namespace std;
using namespace cv;

double margen = 0.8;

struct Kalman {
	KalmanFilter kf = KalmanFilter(2, 1, 0);
	Mat processNoise = Mat(2, 1, CV_32F);
	Mat measurement = Mat::zeros(1, 1, CV_32F);
	Mat state = Mat(2, 1, CV_32F);

	void init() {
		kf.transitionMatrix = (Mat_<float>(2, 2) << 1, 0, 0, 1);
		setIdentity(kf.measurementMatrix);
		setIdentity(kf.processNoiseCov, Scalar::all(1e-5));
		setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));
		setIdentity(kf.errorCovPost, Scalar::all(1));
	}

	float predict() {
		Mat prediction = kf.predict();
		return prediction.at<float>(0);
	}

	void correct() {
		kf.correct(measurement);
	}

	void update(float a) {
		state.at<float>(1) = 0;
		state.at<float>(0) = a;
		kf.statePost = state;
		randn(measurement, Scalar::all(0), Scalar::all(kf.measurementNoiseCov.at<float>(0)));
		measurement += kf.measurementMatrix*state;

	}


};

struct Stabilizer {
	Mat gray, image, prevGray;
	vector<Point2f> pt[2];
	Mat res;
	vector<uchar> status;
	vector<float> errors;
	Mat_<float> t;
	vector<Point2f> f[2];
	float xa, ya, ta;

	Kalman k = Kalman();

	bool displayFlow, preventTranslation, preventScale, preventRotation, doKalman;

	Stabilizer() {
		displayFlow, preventTranslation,preventScale = false;
		t = Mat::eye(3, 3, CV_32FC1);
		k.init();
	}

	Stabilizer(bool f) : displayFlow(f) {
		preventTranslation, preventScale = false;
		t = Mat::eye(3, 3, CV_32FC1);
		k.init();
	}

	Stabilizer(bool f, bool tr, bool sc) : displayFlow(f), preventTranslation(tr), preventScale(sc) {
		t = Mat::eye(3, 3, CV_32FC1);
		k.init();
	}

	void process(Mat &img) {
		cvtColor(img, gray, CV_BGR2GRAY);
		pt[0].clear();
		pt[1].clear();
		goodFeaturesToTrack(gray, pt[0], 400, 0.01, 10);
		f[0].clear();
		f[1].clear();
		status.clear();
		errors.clear();
		if (!prevGray.empty()) {
			calcOpticalFlowPyrLK(prevGray, gray, pt[0], pt[1], status, errors);
			for (int i = 0; i < pt[0].size(); i++) {
				if (status[i] != 0) {
					f[0].push_back(pt[0][i]);
					f[1].push_back(pt[1][i]);
				}
			}
			if (f[0].size() < status.size() * margen) {
				//Error
				cout << "Error de margen" << endl;
				gray.copyTo(prevGray);
				img.copyTo(image);
				return;
			}
			if (f[0].size() == 0) {
				//Error
				cout << "Error, no hay puntos de interes" << endl;
				gray.copyTo(prevGray);
				img.copyTo(image);
				return;
			}
			Mat n = estimateRigidTransform(f[0], f[1], false);
			if (n.rows == 0 && n.cols == 0) {
				//Error
				cout << "Error, no hay transformacion rigida" << endl;
				gray.copyTo(prevGray);
				img.copyTo(image);
				return;
			}
			Mat_<float> nr = Mat_<float>::eye(3, 3);
			n.copyTo(nr.rowRange(0, 2));
			float a = nr.at<float>(Point(0, 0));
			float b = nr.at<float>(Point(0, 1));
			float ang = cvFastArctan(b, a);
			float x = a / cosf(ang* M_PI / 180);
			float y = -nr.at<float>(Point(1, 0)) / sinf(ang * M_PI / 180);

			if (preventTranslation) {

				nr.at<float>(Point(2, 0)) = 0;
				nr.at<float>(Point(2, 1)) = 0;
			}

			if (preventScale) {
				nr.at<float>(Point(0, 0)) /= x;
				nr.at<float>(Point(0, 1)) /= x;
				nr.at<float>(Point(1, 0)) /= y;
				nr.at<float>(Point(1, 1)) /= y;
			}

			float prediction = k.predict();
			k.update(ang);
			k.correct();
			
			prediction = k.measurement.at<float>(0);
			nr.at<float>(Point(0, 0)) = cosf((prediction) * M_PI / 180);
			nr.at<float>(Point(0, 1)) = sinf((prediction) * M_PI / 180);
			nr.at<float>(Point(1, 0)) = -sinf((prediction) * M_PI / 180);
			nr.at<float>(Point(1, 1)) = cosf((prediction) * M_PI / 180);

			t *= nr;
		}

		gray.copyTo(prevGray);
		img.copyTo(image);
	}

	Mat getImg() {
		return getImgUsingRigid();
	}

	Mat getImgUsingRigid() {
		Mat i = t.inv();
		Mat w;
		warpAffine(image, w, i.rowRange(0, 2), Size());
		return w;
	}

	Mat getOriginal() {
		Mat o;
		image.copyTo(o);
		if (displayFlow) {
			for (int k = 0; k < f[0].size(); k++)
			{
				circle(o, f[0][k], 1, Scalar(155), -1, 8);
				line(o, f[0][k], f[1][k], Scalar(255));
				circle(o, f[1][k], 2, Scalar(255), -1, 8);
			}
		}
		return o;
	}
};


int main(int argc, char **args)
{	
	VideoCapture video = VideoCapture("video.mp4");
	Mat frame;
	bool next = true;
	int fps;
	clock_t start, end;
	
	//Primer bool: mostrar flujo optico
	//Segundo bool: eliminar la traslacion dentro de la matriz
	//Tercer bool: eliminar el escalamiento dentro de la matriz
	Stabilizer f = Stabilizer(true, true, true);
	int i = 1;
	namedWindow("stabilized", WINDOW_AUTOSIZE);
	while (next) {
		start = clock();
		next = video.read(frame);		
		if (next) {
			
			i++;
			if (frame.cols > 1200) {
				resize(frame, frame, Size(frame.cols / 5, frame.rows / 5));
			}
			f.process(frame);

			
			end = clock();
			double diffticks = end - start;
			double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
			fps = 1000 / diffms;
			string fp = to_string(fps);
			
			
			Mat pr = f.getImg();

			putText(pr, fp, Point(0, pr.rows - 20), HersheyFonts::FONT_HERSHEY_PLAIN, 4, Scalar(255, 255, 255), 5);

			imshow("video", f.getOriginal());

			imshow("stabilized", pr);
			
			if (waitKey(1) >= 0) {
				break;
			}
			
		}

	}

	video.release();

	waitKey();

	return 0;
}