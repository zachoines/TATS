#include "ArucoDetector.h"
#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "../util/util.h"

namespace Detect {
	
	std::vector<struct DetectionData> ArucoDetector::detect(cv::Mat& image) {
        _allCorners.clear();
        _ids.clear();
		std::vector<struct DetectionData> detectionResults;
        cv::aruco::detectMarkers(image, _dictionary, _allCorners, _ids);

        if (_ids.size() > 0) {
            for(std::vector<cv::Point2f> corners: _allCorners) {
                cv::Point2f center(0.f, 0.f);

                for(cv::Point2f corner : corners) {
                    center += corner;
                }
                
                center /= 4.0;

                cv::RotatedRect rc = cv::minAreaRect(corners);
                cv::Rect2i box = rc.boundingRect();

                struct DetectionData detection = { 
					.confidence = 1.0, 
					.found = true, 
					.label = 0, 
					.target = "ArUco", 
					.center = cv::Point2i(center), 
					.boundingBox = box
				};
				
				detectionResults.push_back(detection);
            }
        }
        
		return detectionResults;
	}

	std::vector<struct DetectionData> ArucoDetector::detect(cv::Mat& image, bool draw, int numDrawn)
	{
		std::vector<struct DetectionData> detectionResults = this->detect(image);

        // if (draw) {
        //     cv::aruco::drawDetectedMarkers(image, _allCorners);
        // }

        if (draw && !detectionResults.empty()) {
			int count = 0;
			for (struct DetectionData detection : detectionResults) {
				if (detection.found && count <= numDrawn) {
					count++;
					Utility::drawPred(detection.confidence, detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.x + detection.boundingBox.width, detection.boundingBox.y + detection.boundingBox.height, image, detection.target);
				}
			}
			
		}
        
		return detectionResults;
	}

	ArucoDetector::ArucoDetector(int dict)
	{ 
        _dictionary = cv::aruco::getPredefinedDictionary(dict);
	}
}