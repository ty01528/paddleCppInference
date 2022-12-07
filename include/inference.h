//
// Created by ty on 2022/11/29.
//

#ifndef CLAS_SYSTEM_INFERENCE_H
#define CLAS_SYSTEM_INFERENCE_H

#include "include/cls.h"
#include "include/cls_config.h"

using namespace PaddleClas;

namespace Inference {
    class clsInference {
    public:
        void clsInferenceRun(Classifier &classifier, const cv::Mat& img) const;

        Classifier clsInferenceInit(const string& yaml_path);
    private:
        int config_id_map_size = 0;
        int config_topk = 0;
    };
};


#endif //CLAS_SYSTEM_INFERENCE_H
