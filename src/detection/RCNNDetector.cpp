#include "RCNNDetector.h"
#include "../util/util.h"

// C++
#include <string>
#include <vector>

// Pytorch
#include <torch/torch.h>
#include <torch/script.h>

// OpenCV
#include <opencv4/opencv2/ximgproc/segmentation.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

namespace Detect {
    RCNNDetector::RCNNDetector(std::string path) {
        if (Utility::fileExists(path)) {
            
            // Load pytorch model
            try {
                module = torch::jit::load(path);

                if (torch::cuda::is_available()) {
					module.eval();
                    device = torch::kCUDA;
                    module.to(device);
                }

            } catch (const c10::Error &e) {
                throw std::runtime_error("Error loading the model.");
            }
        } else {
            throw std::runtime_error("Could not load model from file " + path);
        }
    }

    void RCNNDetector::setTargetLabel(int label) {
        this->targetLabel = label;
    }

    std::vector<struct DetectionData> RCNNDetector::detect(cv::Mat& image) {
        std::vector<struct DetectionData> detectionResults;
		cv::Mat src = image.clone();

        // Perform selective search on image
		cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
		ss->setBaseImage(src);
		ss->switchToSelectiveSearchFast();
		std::vector<cv::Rect> rects;
		ss->process(rects);
		rects.resize(num_rects);

        // Image to correct formate
		cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
		src.convertTo(src, CV_32FC3, 1 / 255.0);

		// For input/out of model
		std::vector<cv::Mat> smallImages;
		std::vector<torch::jit::IValue> net_input;
		std::vector<at::Tensor> inputs_t;
		std::vector<at::Tensor> outputs_t;

		// For NMS
		std::vector<cv::Rect> nmsBoxes;
		std::vector<float> nmsScores;

		// Iterate through proposed ROI's
		for (int i = 0; i < rects.size(); i++) {
			if (i < num_rects) {

				// Massage image to correct formate
				cv::Mat smallimage = cv::Mat(src, rects[i]);
				cv::resize(smallimage, smallimage, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
				smallimage.convertTo(smallimage, CV_32FC3, 1 / 255.0);
				at::Tensor tensor_image = at::from_blob(smallimage.data, { 1, smallimage.rows, smallimage.cols, 3 }, at::kFloat);
				at::Tensor reshaped_src = tensor_image.permute({ 0, 3, 1, 2 });
				
				// Mean/std normalization
				reshaped_src[0][0] = reshaped_src[0][0].sub(0.485).div(0.229);
				reshaped_src[0][1] = reshaped_src[0][1].sub(0.456).div(0.224);
				reshaped_src[0][2] = reshaped_src[0][2].sub(0.406).div(0.225);
				
				// Build input batch
				inputs_t.push_back(reshaped_src.clone());
			}
		}
		
		
		// Pass through pytorch model, find argmax and build up Rects/Scores for NMS
		at::Tensor input_batch = at::cat(inputs_t, 0).to(device);
		net_input.push_back(input_batch);
		at::Tensor output_batch = module.forward(net_input).toTensor();
		at::Tensor labels = output_batch.argmax(-1);

		for (int i = 0; i < num_rects; i++) {
		
			int label = labels[i].item().toInt();
			if (label == targetLabel) {
				nmsScores.push_back(output_batch[i][label].item().toDouble());
				nmsBoxes.push_back(rects[i]);
			}
		} 

		// Apply Non-Max Suppression
		std::vector<int> indices;
		cv::dnn::NMSBoxes(rects, nmsScores, 0.99, 0.5, indices, 1.0, max_detections);

		for (int idx : indices) {
			cv::Rect box = rects[idx];

			int objX = box.width / 2;
			int objY = box.height / 2;

			struct DetectionData detection = { 
				.confidence = nmsScores[idx], 
				.found = true, 
				.label = this->targetLabel,
				.target = "", 
				.center = cv::Point2i(objX, objY), 
				.boundingBox = box 
			};

			detectionResults.push_back(detection);

		}

		
        return detectionResults;
    }
    
    std::vector<struct DetectionData> RCNNDetector::detect(cv::Mat& image, bool draw, int numDrawn) {
        
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
};