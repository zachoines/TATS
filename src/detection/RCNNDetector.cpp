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

    struct DetectionData RCNNDetector::detect(cv::Mat& image) {
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
		std::vector<torch::Tensor> inputs_t;
		std::vector<torch::Tensor> outputs_t;
		at::Tensor reshaped_src;

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
				at::Tensor tensor_image = torch::from_blob(smallimage.data, { 1, smallimage.rows, smallimage.cols, 3 }, at::kFloat);
				reshaped_src = tensor_image.permute({ 0, 3, 1, 2 });
				
				// Mean/std normalization
				reshaped_src[0][0] = reshaped_src[0][0].sub(0.485).div(0.229);
				reshaped_src[0][1] = reshaped_src[0][1].sub(0.456).div(0.224);
				reshaped_src[0][2] = reshaped_src[0][2].sub(0.406).div(0.225);
				
				// Build input batch
				inputs_t.push_back(reshaped_src.clone());
			}
		}
		
		
		// Pass through pytorch model, find argmax and build up Rects/Scores for NMS
		torch::Tensor input_batch = torch::cat(inputs_t, 0);
		net_input.push_back(input_batch.to(device));
		torch::Tensor output_batch = module.forward(net_input).toTensor();
		torch::Tensor labels = output_batch.argmax(-1);
		std::cout << output_batch << std::endl;
		std::cout << labels << std::endl;
		for (int i = 0; i < num_rects; i++) {
		
			int label = labels[i].item().toInt();
			if (label == targetLabel) {
				nmsScores.push_back(output_batch[i][label].item().toDouble());
				nmsBoxes.push_back(rects[i]);
			}
		} 

		// Apply Non-Max Suppression
		std::vector<int> indices;
		cv::dnn::NMSBoxes(rects, nmsScores, 0.99, 0.5, indices, 1.0, 1);
		int idx = indices.at(0);
		cv::Rect box = rects[idx];

        int objX = box.width / 2;
        int objY = box.height / 2;

        this->last_frame = &src;
        struct DetectionData detectionResults = { .targetCenterX = objX, .targetCenterY = objY, .confidence = nmsScores[idx], .found = true, .target = "", .label = this->targetLabel, .boundingBox = box };
        return detectionResults;
    }
    
    struct DetectionData RCNNDetector::detect(cv::Mat& src, bool draw) {
        
        struct DetectionData detectionResults = this->detect(src);
        if (draw) {
            Utility::drawPred(detectionResults.confidence, detectionResults.boundingBox.x, detectionResults.boundingBox.y, detectionResults.boundingBox.x + detectionResults.boundingBox.width, detectionResults.boundingBox.y + detectionResults.boundingBox.height, src);
        }
    }
};