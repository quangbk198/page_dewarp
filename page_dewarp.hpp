#ifndef PAGE_DEWARP_HPP
#define PAGE_DEWARP_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

// #define PAGE_MARGIN_X 8 // reduced px to ignore near L/R edge
// #define PAGE_MARGIN_Y 8 //reduced px to ignore near T/B edge

#define OUTPUT_ZOOM 1.0   // how much to zoom output relative to *original* image
#define REMAP_DECIMATE 16 // downscaling factor for remapping image

// #define ADAPTIVE_WINSZ 55 // window size for adaptive threshold in reduced px

// #define TEXT_MIN_WIDTH 15     // min reduced px width of detected text contour
// #define TEXT_MIN_HEIGHT 5     // min reduced px height of detected text contour
// #define TEXT_MIN_ASPECT 1.5   // filter out text contours below this w/h ratio
// #define TEXT_MAX_THICKNESS 10 // max reduced px thickness of detected text contour

// #define EDGE_MAX_OVERLAP 1.0  // max reduced px horiz. overlap of contours in span
// #define EDGE_MAX_LENGTH 100.0 // max reduced px length of edge connecting contours
// #define EDGE_ANGLE_COST 10.0  // cost of angles in edges (tradeoff vs. length)
// #define EDGE_MAX_ANGLE 7.5    // maximum change in angle allowed between contours

// #define slice(src, a, b) src.begin() + a, src.begin() + b - 1

#define SPAN_MIN_WIDTH 40   // minimum reduced px width for span
#define SPAN_PX_PER_STEP 40 // reduced px spacing for sampling along spans
#define FOCAL_LENGTH 1.2    // normalized focal length of camera

#define DEBUG_LEVEL 2      // 0=none, 1=only result remap, 2=all
#define DEBUG_OUTPUT "file" // file, screen, both

#define WINDOW_NAME "Dewarp" // Window name for visualization

#define MASK_TYPE_TEXT 0
#define MASK_TYPE_LINE 1

using namespace std;
using namespace cv;

class ContourInfo {
    public:
        double point0[2];
        double point1[2];
        double center[2];
        double local_xrng[2];
        double tangent[2];
        double angle;
        vector<cv::Point> contour;
        cv::Rect rect;
        cv::Mat mask; 
        ContourInfo *pred, *succ;       //predecessor and successors

        ContourInfo();
        ContourInfo(const ContourInfo &c);
        ContourInfo(vector<cv::Point> c, cv::Rect r, cv::Mat m);

        double project_x (cv::Point point);     //tìm hình chiếu của một điểm lên một đường thẳng
        double interval_measure_overlap (double *int_a, double *int_b);
        double local_overlap (ContourInfo other);

        bool operator == (const ContourInfo &ci0) const;
        void operator = (const ContourInfo &c);
};

class Edge {
    public:
        double score;
        ContourInfo *cinfo_a;
	    ContourInfo *cinfo_b;

        Edge() {
            score = 0;
            cinfo_a = NULL;
		    cinfo_b = NULL;
        }

        Edge(double s, ContourInfo *ci_a, ContourInfo *ci_b)
        {
            this->score = s;
            this->cinfo_a = ci_a;
            this->cinfo_b = ci_b;
        }
};

class Optimize {
    public:
        void make_keypoint_index(std::vector<int> span_counts);

        double Minimize(std::vector<double> params);

        void remap_image(string name, 
                        cv::Mat img, 
                        cv::Mat small, 
                        cv::Mat &thresh, 
                        std::vector<double> page_dims, 
                        std::vector<double> params, 
                        string outfile_prefix); 
};

#endif