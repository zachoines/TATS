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
            } catch (...) {
                throw std::runtime_error("Error loading Yolo5 model");
            }
    
        } else {
            throw std::runtime_error("Could not load model from file " + path);
        }
    }

    std::vector<struct DetectionData> YoloDetector::detect(cv::Mat& image) {
        try {
            return run(image);
        } catch (...) {
            throw std::runtime_error("Issue running inferance on image");
        }
    }

    std::vector<struct DetectionData> YoloDetector::detect(cv::Mat& image, bool draw, int numDrawn) {
        std::vector<struct DetectionData> detections = detect(image);

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

    void YoloDetector::setTargets(std::vector<int> targetLabels) {
        this->targetLabels = targetLabels;
    }

    void YoloDetector::setConfidence(float conf_threshold) {
        this->conf_threshold = conf_threshold;
    }

    void YoloDetector::setIOU(float iou_threshold) {
        this->iou_threshold = iou_threshold;
    }

    /** Private Class Methods **/
    void YoloDetector::loadModule(const std::string& model_path) {
        try {
            module_ = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            throw std::runtime_error("Error loading the model");
        }

        half_ = (device_ != torch::kCPU);
        module_.to(device_);

        if (half_) {
            module_.to(torch::kHalf);
        }

        module_.eval();
    }


    std::vector<DetectionData> YoloDetector::run(const cv::Mat& img) {
        torch::NoGradGuard no_grad;

        // keep the original image for visualization purpose
        cv::Mat img_input = img.clone();
        std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
        const float pad_w = pad_info[0];
        const float pad_h = pad_info[1];
        const float scale = pad_info[2];


        /*** Pre-Process ***/
        cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
        img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
        auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

        tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

        if (half_) {
            tensor_img = tensor_img.to(torch::kHalf);
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(tensor_img);

        /*** Inference ***/
        torch::jit::IValue output = module_.forward(inputs);

        /*** Post-process ***/
        torch::Tensor detections = output.toTuple()->elements()[0].toTensor();

        // result: n * 7
        // batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
        torch::Tensor result = PostProcessing(detections);

        // Note - only the first image in the batch will be used for demo
        auto idx_mask = result * (result.select(1, 0) == 0).to(torch::kFloat32).unsqueeze(1);
        auto idx_mask_index =  torch::nonzero(idx_mask.select(1, 1)).squeeze();
        const auto& data = result.index_select(0, idx_mask_index).slice(1, 1, 7);

        // use accessor to access tensor elements efficiently
        // remap to original image and list bounding boxes for debugging purpose
        std::vector<DetectionData> det = ScaleCoordinates(data.accessor<float, 2>(), pad_w, pad_h, scale, img.size());

        return det;
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

        cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
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


    torch::Tensor YoloDetector::PostProcessing(const torch::Tensor& detections) {
        constexpr int item_attr_size = 5;
        int batch_size = detections.size(0);
        auto num_classes = detections.size(2) - item_attr_size;  // 80 for coco dataset

        // get candidates which object confidence > threshold
        auto conf_mask = detections.select(2, 4).ge(conf_threshold).unsqueeze(2);

        // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
        detections.slice(2, item_attr_size, item_attr_size + num_classes) *=
                detections.select(2, 4).unsqueeze(2);

        // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
        torch::Tensor box = torch::zeros(detections.sizes(), detections.options());
        box.select(2, Det::tl_x) = detections.select(2, 0) - detections.select(2, 2).div(2);
        box.select(2, Det::tl_y) = detections.select(2, 1) - detections.select(2, 3).div(2);
        box.select(2, Det::br_x) = detections.select(2, 0) + detections.select(2, 2).div(2);
        box.select(2, Det::br_y) = detections.select(2, 1) + detections.select(2, 3).div(2);
        detections.slice(2, 0, 4) = box.slice(2, 0, 4);

        bool is_initialized = false;
        torch::Tensor output = torch::zeros({0, 7});

        // iterating all images in the batch
        for (int batch_i = 0; batch_i < batch_size; batch_i++) {
            auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

            // if none remain then process next image
            if (det.size(0) == 0) {
                continue;
            }

            // get the max classes score at each result (e.g. elements 5-84)
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

            // class score
            auto max_conf_score = std::get<0>(max_classes);
            // index
            auto max_conf_index = std::get<1>(max_classes);

            max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);
            max_conf_index = max_conf_index.to(torch::kFloat32).unsqueeze(1);

            // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
            det = torch::cat({det.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

            // get unique classes
            std::vector<torch::Tensor> img_classes;

            auto len = det.size(0);
            for (int i = 0; i < len; i++) {
                bool found = false;
                for (const auto& cls : img_classes) {
                    auto ret = (det[i][Det::class_idx] == cls);
                    if (torch::nonzero(ret).size(0) > 0) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    img_classes.emplace_back(det[i][Det::class_idx]);
                }
            }

            // iterating all unique classes
            for (const auto& cls : img_classes) {
                auto cls_mask = det * (det.select(1, Det::class_idx) == cls).to(torch::kFloat32).unsqueeze(1);
                auto class_mask_index =  torch::nonzero(cls_mask.select(1, Det::score)).squeeze();
                auto bbox_by_class = det.index_select(0, class_mask_index).view({-1, 6});

                // sort by confidence (descending)
                std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(bbox_by_class.select(1, 4), -1, true);
                auto conf_sort_index = std::get<1>(sort_ret);

                bbox_by_class = bbox_by_class.index_select(0, conf_sort_index.squeeze()).cpu();
                int num_by_class = bbox_by_class.size(0);

                // Non-Maximum Suppression (NMS)
                for(int i = 0; i < num_by_class - 1; i++) {
                    auto iou = GetBoundingBoxIoU(bbox_by_class[i].unsqueeze(0), bbox_by_class.slice(0, i + 1, num_by_class));
                    auto iou_mask = (iou < iou_threshold).to(torch::kFloat32).unsqueeze(1);

                    bbox_by_class.slice(0, i + 1, num_by_class) *= iou_mask;

                    // remove from list
                    auto non_zero_index = torch::nonzero(bbox_by_class.select(1, 4)).squeeze();
                    bbox_by_class = bbox_by_class.index_select(0, non_zero_index).view({-1, 6});
                    // update remain number of detections
                    num_by_class = bbox_by_class.size(0);
                }

                torch::Tensor batch_index = torch::zeros({bbox_by_class.size(0), 1}).fill_(batch_i);

                if (!is_initialized) {
                    output = torch::cat({batch_index, bbox_by_class}, 1);
                    is_initialized = true;
                }
                else {
                    auto out = torch::cat({batch_index, bbox_by_class}, 1);
                    output = torch::cat({output,out}, 0);
                }
            }
        }

        return output;
    }


    std::vector<struct DetectionData> YoloDetector::ScaleCoordinates(const at::TensorAccessor<float, 2>& data,
                                                    float pad_w, float pad_h, float scale, const cv::Size& img_shape) {
        auto clip = [](float n, float lower, float upper) {
            return std::max(lower, std::min(n, upper));
        };

        std::vector<struct DetectionData> detections;
        for (int i = 0; i < data.size(0) ; i++) {
    
            float x1 = (data[i][Det::tl_x] - pad_w)/scale;  // x padding
            float y1 = (data[i][Det::tl_y] - pad_h)/scale;  // y padding
            float x2 = (data[i][Det::br_x] - pad_w)/scale;  // x padding
            float y2 = (data[i][Det::br_y] - pad_h)/scale;  // y padding

            x1 = clip(x1, 0, img_shape.width);
            y1 = clip(y1, 0, img_shape.height);
            x2 = clip(x2, 0, img_shape.width);
            y2 = clip(y2, 0, img_shape.height);

            int centerX = x1 + img_shape.width * 0.5;
            int centerY = y1 + img_shape.height * 0.5;

            struct DetectionData detection = { 
                .confidence = data[i][Det::score], 
                .found = true, 
                .label = data[i][Det::class_idx], 
                .target = "", 
                .center = cv::Point2i(centerX, centerY), 
                .boundingBox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)) 
            };

            detections.emplace_back(detection);
        }

        return detections;
    }
};
