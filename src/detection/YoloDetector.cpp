#include "YoloDetector.h"

namespace Detect {

    YoloDetector::~YoloDetector() {
        
    }

    YoloDetector::YoloDetector(std::string path, std::vector<std::string> classNames) {
        if (Utility::fileExists(path)) {
            this->class_names = classNames;
            if (torch::cuda::is_available()) {
                device_ = torch::kCUDA;
            }
        
            try {
                loadModule(path);
                cv::Mat img(640, 640, CV_8UC3);
                cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
                run(img);
            } catch (c10::Error e) {
                std::cerr << e.what() << std::endl;
                throw std::runtime_error("Error loading Yolo5 model");
            }
    
        } else {
            throw std::runtime_error("Could not load model from file " + path);
        }
    }

    std::vector<DetectionData> YoloDetector::detect(cv::Mat& image) {
        try {
            std::vector<std::vector<DetectionData>> detections = run(image);
            std::vector<DetectionData> detection;
            
            if (!detections.empty()) {
                detection = detections[0];
              
            } 

            return detection;
        } catch (c10::Error e) {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Issue running inferance on image");
        }
    }

    std::vector<DetectionData> YoloDetector::detect(cv::Mat& image, bool draw, int numDrawn) {
        std::vector<DetectionData> detections = detect(image);

        if (draw and !detections.empty()) {
            for (const auto& detection : detections) {
                const auto& box = detection.boundingBox;
                float score = detection.confidence;
                int class_idx = detection.label;

                cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2);

                
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);

                cv::rectangle(image,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(image, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
                
            }
        }

        return detections;
    }

    std::vector<std::vector<DetectionData>> YoloDetector::run(const cv::Mat& img) {
        torch::NoGradGuard no_grad;

        // keep the original image for visualization purpose
        cv::Mat img_input = img.clone();

        std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
        const float pad_w = pad_info[0];
        const float pad_h = pad_info[1];
        const float scale = pad_info[2];

        // Formate image correctly. RGB, 0 to 1, BCHW
        cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  
        img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f); 
        auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);
        tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous(); 

        if (half_) {
            tensor_img = tensor_img.to(torch::kHalf);
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(tensor_img);
        torch::jit::IValue output = module_.forward(inputs);
    
        // result: n * 7
        // batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
        return PostProcessing(output.toTuple()->elements()[0].toTensor(), pad_w, pad_h, scale, img.size(), conf_threshold, iou_threshold);
    }

    // returns the IoU of bounding boxes
    torch::Tensor YoloDetector::GetBoundingBoxIoU(const torch::Tensor& box1, const torch::Tensor& box2) {
        // get the coordinates of bounding boxes
        const torch::Tensor& b1_x1 = box1.select(1, 0);
        const torch::Tensor& b1_y1 = box1.select(1, 1);
        const torch::Tensor& b1_x2 = box1.select(1, 2);
        const torch::Tensor& b1_y2 = box1.select(1, 3);

        const torch::Tensor& b2_x1 = box2.select(1, 0);
        const torch::Tensor& b2_y1 = box2.select(1, 1);
        const torch::Tensor& b2_x2 = box2.select(1, 2);
        const torch::Tensor& b2_y2 = box2.select(1, 3);

        // get the coordinates of the intersection rectangle
        torch::Tensor inter_rect_x1 =  torch::max(b1_x1, b2_x1);
        torch::Tensor inter_rect_y1 =  torch::max(b1_y1, b2_y1);
        torch::Tensor inter_rect_x2 =  torch::min(b1_x2, b2_x2);
        torch::Tensor inter_rect_y2 =  torch::min(b1_y2, b2_y2);

        // calculate intersection area
        torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1,torch::zeros(inter_rect_x2.sizes()))
                                * torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

        // calculate union area
        torch::Tensor b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1);
        torch::Tensor b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1);

        // calculate IoU
        torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

        return iou;
    }


    std::vector<std::vector<DetectionData>> YoloDetector::PostProcessing(const torch::Tensor& detections, float pad_w, float pad_h, float scale, const cv::Size& img_shape, float conf_thres, float iou_thres) {
        constexpr int item_attr_size = 5;
        int batch_size = detections.size(0);
        auto num_classes = detections.size(2) - item_attr_size;

        // get candidates which object confidence > threshold
        auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

        std::vector<std::vector<DetectionData>> output;
        output.reserve(batch_size);

        // iterating all images in the batch
        for (int batch_i = 0; batch_i < batch_size; batch_i++) {
            // apply constrains to get filtered detections for current image
            auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

            // if none detections remain then skip and start to process next image
            if (0 == det.size(0)) {
                continue;
            }

            // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
            det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

            // box (center x, center y, width, height) to (x1, y1, x2, y2)
            torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

            // [best class only] get the max classes score at each result (e.g. elements 5-84)
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

            // class score
            auto max_conf_score = std::get<0>(max_classes);
            
            // index
            auto max_conf_index = std::get<1>(max_classes);

            max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
            max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

            // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
            det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

            // for batched NMS
            constexpr int max_wh = 4096;
            auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
            auto offset_box = det.slice(1, 0, 4) + c;

            std::vector<cv::Rect> offset_box_vec;
            std::vector<float> score_vec;

            // copy data back to cpu
            auto offset_boxes_cpu = offset_box.cpu();
            auto det_cpu = det.cpu();
            const auto& det_cpu_array = det_cpu.accessor<float, 2>();

            // use accessor to access tensor elements efficiently
            Tensor2Detection(offset_boxes_cpu.accessor<float,2>(), det_cpu_array, offset_box_vec, score_vec);

            // run NMS
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

            std::vector<DetectionData> det_vec;
            for (int index : nms_indices) {
                DetectionData t;
                const auto& b = det_cpu_array[index];
                t.boundingBox = cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]), cv::Point(b[Det::br_x], b[Det::br_y])); 
                t.confidence = det_cpu_array[index][Det::score];
                t.label = det_cpu_array[index][Det::class_idx];
                t.target = class_names[t.label];
                det_vec.emplace_back(t);

                if (t.confidence > conf_threshold) {
                    t.found = true;
                }
            }

            ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);

            // save final detection for the current image
            output.emplace_back(det_vec);
        } // end of batch iterating

        return output;
    }


    void YoloDetector::ScaleCoordinates(std::vector<DetectionData>& data,float pad_w, float pad_h, float scale, const cv::Size& img_shape) {
        auto clip = [](float n, float lower, float upper) {
            return std::max(lower, std::min(n, upper));
        };

        std::vector<DetectionData> detections;
        for (auto & i : data) {
            float x1 = (i.boundingBox.tl().x - pad_w)/scale;  // x padding
            float y1 = (i.boundingBox.tl().y - pad_h)/scale;  // y padding
            float x2 = (i.boundingBox.br().x - pad_w)/scale;  // x padding
            float y2 = (i.boundingBox.br().y - pad_h)/scale;  // y padding

            x1 = clip(x1, 0, img_shape.width);
            y1 = clip(y1, 0, img_shape.height);
            x2 = clip(x2, 0, img_shape.width);
            y2 = clip(y2, 0, img_shape.height);

            i.boundingBox = cv::Rect2d(cv::Point2d(x1, y1), cv::Point2d(x2, y2));
            i.center = (i.boundingBox.tl() + i.boundingBox.br()) / 2.0;
        }
    }

    torch::Tensor YoloDetector::xywh2xyxy(const torch::Tensor& x) {
        auto y = torch::zeros_like(x);
        // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
        y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
        y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
        y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
        y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
        return y;
    }

    std::vector<float> YoloDetector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
        auto in_h = static_cast<float>(src.rows);
        auto in_w = static_cast<float>(src.cols);
        float out_h = out_size.height;
        float out_w = out_size.width;

        float scale = std::min(out_w / in_w, out_h / in_h);

        int mid_h = static_cast<int>(in_h * scale);
        int mid_w = static_cast<int>(in_w * scale);

        cv::resize(src, dst, cv::Size(mid_w, mid_h));

        int top = (static_cast<int>(out_h) - mid_h) / 2;
        int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
        int left = (static_cast<int>(out_w)- mid_w) / 2;
        int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

        cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
    }
    
    void YoloDetector::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes, const at::TensorAccessor<float, 2>& det, std::vector<cv::Rect>& offset_box_vec, std::vector<float>& score_vec) {

        for (int i = 0; i < offset_boxes.size(0) ; i++) {
            offset_box_vec.emplace_back(
                    cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                            cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
            );
            score_vec.emplace_back(det[i][Det::score]);
        }
    }

    void YoloDetector::setTargets(std::vector<int> targetLabels) {
        this->targetLabels = targetLabels;
    }

    void YoloDetector::setConfidence(float conf_threshold) {
        this->conf_threshold = conf_threshold;
    }

    void YoloDetector::setIOU(float iou_threshold) {
        this->iou_threshold = iou_threshold;
    }

    void YoloDetector::loadModule(const std::string& model_path) {
        try {
            module_ = torch::jit::load(model_path);
        }
        catch (c10::Error e) {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Error loading the model");
        }

        half_ = (device_ != torch::kCPU);
        module_.to(device_);

        if (half_) {
            module_.to(torch::kHalf);
        }

        module_.eval();
    }
};
