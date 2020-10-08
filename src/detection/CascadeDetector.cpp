#include "CascadeDetector.h"
#include <vector>
#include <string>
// #include "opencv2/objdetect.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 

namespace TATS {

	DetectionData CascadeDetector::detect(cv::Mat& src, std::string target)
	{

		this->last_frame = &src;
		std::vector<cv::Rect> faces;

		if (input_color) {
			// Detect faces of different sizes using cascade classifier  
			this->cascade.detectMultiScale(src, faces, 1.3,
				2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
			
		}
		else {
			cv::Mat gray, smallImg;

			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY); // Convert to Gray Scale 
			double fx = 1 / scale;

			// Resize the Grayscale Image  
			cv:resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
			cv::equalizeHist(smallImg, smallImg);

			// Detect faces of different sizes using cascade classifier  
			this->cascade.detectMultiScale(smallImg, faces, 1.3,
				2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
		}

		if (!faces.empty() && !faces[0].empty()) {

			int x1 = faces[0].x;
			int y1 = faces[0].y;

			int width = faces[0].width;
			int height = faces[0].height;

			float centerX = x1 + width * 0.5;
			float centerY = y1 + height * 0.5;

			struct Rect objectBounds = { .x = x1, .y = y1, .height = height , .width = width };
			struct DetectionData detectionResults = { .targetCenterX = centerX, .targetCenterY = centerY, .confidence = 1.0, .found = true, .target = "face", .boundingBox = objectBounds };
			return detectionResults;
		}
		

		struct DetectionData detectionResults;
		detectionResults.found = false;
		return detectionResults;
	}

	// Draws bounding box and center circle onto frame. Good for debugging.
	DetectionData CascadeDetector::detect(cv::Mat& src, std::string target, bool draw)
	{
		struct DetectionData detectionResults = this->detect(src, "face");
		if (draw) {
			if (detectionResults.found) {
				cv::Scalar color = cv::Scalar(255);
				cv::Rect rec(
					detectionResults.boundingBox.x,
					detectionResults.boundingBox.y,
					detectionResults.boundingBox.width,
					detectionResults.boundingBox.height
				);
				circle(
					src,
					cv::Point(detectionResults.targetCenterX, detectionResults.targetCenterY),
					(int)(detectionResults.boundingBox.width + detectionResults.boundingBox.height) / 2 / 10,
					color, 2, 8, 0);
				rectangle(src, rec, color, 2, 8, 0);
				putText(
					src,
					"face",
					cv::Point(detectionResults.boundingBox.x, detectionResults.boundingBox.y - 5),
					cv::FONT_HERSHEY_SIMPLEX,
					1.0,
					color, 2, 8, 0);
			}
		}

		return detectionResults;
	}


	CascadeDetector::CascadeDetector(std::string path)
	{


		// Change path before execution  
		this->cascade.load(path);

		if (this->cascade.empty()) {
			throw "Cascade detector could not be initialized.Check if haar cascade file is in root directory.";
		}
	}
}