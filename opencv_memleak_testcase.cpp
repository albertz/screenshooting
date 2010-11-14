#include <cv.h>


int main() {
	while(true) {
		int hist_size[] = {40};
		float range[] = {0.0f,255.0f};
		float* ranges[] = {range};
		CvHistogram* hist = cvCreateHist(1, hist_size, CV_HIST_ARRAY, ranges, 1);
		cvReleaseHist(&hist);
	}
}
