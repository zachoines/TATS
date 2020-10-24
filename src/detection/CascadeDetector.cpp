#include "CascadeDetector.h"
#include <vector>
#include <string>
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "../util/util.h"

namespace Detect {
	
	// 0 is color; 1 is gray. Grey by default.
	void CascadeDetector::setInputColor(int code)
	{
		if (code <= 1) {
			input_color = code;
		}
	}

	std::vector<struct DetectionData> CascadeDetector::detect(cv::Mat& image) {
		std::vector<cv::Rect> faces;
		std::vector<struct DetectionData> detectionResults;

		if (this->input_color) {
			
			// Detect faces of different sizes using cascade classifier  
			this->cascade.detectMultiScale(image, faces, 1.3,
				2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
			
		}
		else {
			cv::Mat gray, smallImg;

			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); // Convert to Gray Scale 
			// double fx = 1 / scale;

			// // Resize the Grayscale Image  
			// cv:resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
			// cv::equalizeHist(smallImg, smallImg);

			// Detect faces of different sizes using cascade classifier  
			this->cascade.detectMultiScale(gray, faces, 1.3,
				2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
		}

		if (!faces.empty()) {
			for (cv::Rect face : faces) {

				int x1 = face.x;
				int y1 = face.y;

				int width = face.width;
				int height = face.height;

				if (width <= 1 || height <= 1) {
					std::cout << "Bad detection from cascade detector. Check settings" << std::endl;
					continue;
				}

				int centerX = x1 + width * 0.5;
				int centerY = y1 + height * 0.5;

				struct DetectionData detection = { 
					.confidence = 1.0, 
					.found = true, 
					.label = 0, 
					.target = "face", 
					.center = cv::Point2i(centerX, centerY), 
					.boundingBox = face 
				};
				detectionResults.push_back(detection);
			}
		}
			
		return detectionResults;
		
	}

	// Draws bounding box and center circle onto frame. Good for debugging.
	std::vector<struct DetectionData> CascadeDetector::detect(cv::Mat& image, bool draw, int numDrawn)
	{
		std::vector<struct DetectionData> detectionResults = this->detect(image);

		
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


	CascadeDetector::CascadeDetector(std::string path)
	{ 
		this->cascade.load(path);

		if (this->cascade.empty()) {
			throw std::runtime_error("Cascade detector could not be initialized.Check if haar cascade file is in root directory.");
		}
	}
}