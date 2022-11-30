//
// Created by ty on 2022/11/30.
//

#ifndef CLAS_SYSTEM_INFERENCEINIT_H
#define CLAS_SYSTEM_INFERENCEINIT_H


namespace inferenceInit {
    class clsInferenceInit {
    public:
        void Run(int argc, char **argv);

        void Init(int argc, char **argv);
    };
    class segInferenceInit {
    public:
        void Run(int argc, char **argv);

        void Init(int argc, char **argv);
    };
    class detInferenceInit {
    public:
        void Run(int argc, char **argv);

        void Init(int argc, char **argv);
    };
};


#endif //CLAS_SYSTEM_INFERENCEINIT_H
