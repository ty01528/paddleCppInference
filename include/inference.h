//
// Created by ty on 2022/11/29.
//

#ifndef CLAS_SYSTEM_INFERENCE_H
#define CLAS_SYSTEM_INFERENCE_H


namespace Inference {
    class clsInference {
    public:
        static void Run(PaddleClas::ClsConfig &config, const string& img_path);

        static PaddleClas::ClsConfig Init(int argc, char **argv);
    };
};


#endif //CLAS_SYSTEM_INFERENCE_H
