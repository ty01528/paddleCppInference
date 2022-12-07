//
// Created by ty on 2022/11/29.
//

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <ostream>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include <auto_log/autolog.h>
#include <gflags/gflags.h>
#include <include/cls.h>
#include <include/cls_config.h>
#include <include/inference.h>

using namespace std;
using namespace cv;
using namespace PaddleClas;

namespace Inference{
    Classifier clsInference::clsInferenceInit(const string& yaml_path) {
        std::cout << "starting read configfile" << std::endl;
        ClsConfig config(yaml_path);
        std::cout << "read config complete" << std::endl;
//      config读取完毕
        config.PrintConfigInfo();
        this->config_id_map_size = config.id_map.size();
        this->config_topk = config.topk;
        return Classifier(config);
    }
    void clsInference::clsInferenceRun(Classifier& classifier, const cv::Mat& img) const
    {
        if (img.empty())
        {
            // Handle empty image
        }

        std::cout << "starting decode" << std::endl;

        std::vector<double> classification_times = {0, 0, 0};
        std::vector<double> cls_times_total = {0, 0, 0};
        double infer_time;
        std::vector<float> out_data;
        std::vector<int> result_index;
        bool labelOutputEqualFlag = true;

        // Convert image to RGB color space
        cv::Mat srcimg;
        cv::cvtColor(img, srcimg, cv::COLOR_BGR2RGB);

        classifier.Run(srcimg, out_data, result_index, classification_times);
        if (labelOutputEqualFlag && out_data.size() != this->config_id_map_size)
        {
            std::cout << "Warning: the label size is not equal to output size!"
                      << std::endl;
            labelOutputEqualFlag = false;
        }

        int max_len = std::min(this->config_topk, int(out_data.size()));
//        std::cout << "Current image path: " << img_path << std::endl;
        infer_time = classification_times[0] + classification_times[1] + classification_times[2];
        std::cout << "Current total inferen time cost: " << infer_time << " ms." << std::endl;
        for (int i = 0; i < max_len; ++i)
        {
            printf("\tTop%d: class_id: %d, score: %.4f, ", i + 1, result_index[i], out_data[result_index[i]]);
//            if (label_output_equal_flag)
//                printf("label: %s\n", config.id_map[result_index[i]].c_str());
        }
        std::cout << "\r\n======================end========================\r\n"<< std::endl;

        std::string presion = "fp32";
    }
}