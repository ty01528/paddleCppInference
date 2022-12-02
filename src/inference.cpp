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

DEFINE_string(config,
              "", "Path of yaml file");
DEFINE_string(c,
              "", "Path of yaml file");

namespace Inference{
    Classifier clsInference::Init(const string& yaml_path) {
        std::cout << "starting read configfile" << std::endl;
        ClsConfig config(yaml_path);
        std::cout << "read config complete" << std::endl;
//      config读取完毕
        config.PrintConfigInfo();
        this->config_id_map_size = config.id_map.size();
        this->config_topk = config.topk;
        return Classifier(config);
    }
    void clsInference::Run(Classifier &classifier,const string &img_path){

        if (img_path.empty()) {

        }
        std::cout << "starting decode" << std::endl;

        std::vector<double> cls_times = {0, 0, 0};
        std::vector<double> cls_times_total = {0, 0, 0};
        double infer_time;
        std::vector<float> out_data;
        std::vector<int> result_index;
        bool label_output_equal_flag = true;

//        std::string img_path = img_files_list[idx];
        cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
        if (!srcimg.data) {
            std::cerr << "[ERROR] image read failed! image path: " << img_path
            << "\n";
            exit(-1);
        }

        cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

        classifier.Run(srcimg, out_data, result_index, cls_times);

        if (label_output_equal_flag && out_data.size() != this->config_id_map_size) {
            std::cout << "Warning: the label size is not equal to output size!"
                << std::endl;
            label_output_equal_flag = false;
        }

        int max_len = std::min(this->config_topk, int(out_data.size()));
        std::cout << "Current image path: " << img_path << std::endl;
        infer_time = cls_times[0] + cls_times[1] + cls_times[2];
        std::cout << "Current total inferen time cost: " << infer_time << " ms."
            << std::endl;
        for (int i = 0; i < max_len; ++i) {
            printf("\tTop%d: class_id: %d, score: %.4f, ", i + 1, result_index[i],
                   out_data[result_index[i]]);
//            if (label_output_equal_flag)
//                printf("label: %s\n", config.id_map[result_index[i]].c_str());
        }
        std::cout << "\r\n======================end========================\r\n"<< std::endl;

        std::string presion = "fp32";
//        if (config.use_fp16)
//            presion = "fp16";
//        return 0;
    }
}

