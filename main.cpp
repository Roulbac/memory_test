#include <memory>
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

 
int main(int argc, char * argv[])
{
    Logger gLogger;

    auto builder = nvinfer1::createInferBuilder(gLogger);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile("../model.onnx", static_cast<int>(0));
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(128 * (1 << 20)); // 128 MiB

    auto engine = builder->buildEngineWithConfig(*network, *config);

    builder->destroy();
    network->destroy();
    parser->destroy();
    config->destroy();
    
    for(int i=0; i< atoi(argv[1]); i++)
    {
        auto context = engine->createExecutionContext();
        void* deviceBuffers[2]{0};

        int inputIndex = engine->getBindingIndex("input_rgb:0");
        constexpr int inputNumel = 1 * 128 * 64 * 3;
        int outputIndex = engine->getBindingIndex("truediv:0");
        constexpr int outputNumel = 1 * 128;

        //TODO: Remove batch size hardcoding
        cudaMalloc(&deviceBuffers[inputIndex], 1 * sizeof(float) * inputNumel);
        cudaMalloc(&deviceBuffers[outputIndex], 1 * sizeof(float) * outputNumel);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        float inBuffer[inputNumel] = {0};
        float outBuffer[outputNumel] = {0};

        cudaMemcpyAsync(deviceBuffers[inputIndex], inBuffer, 1 * sizeof(float) * inputNumel, cudaMemcpyHostToDevice, stream);

        context->enqueueV2(deviceBuffers, stream, nullptr);
        
        cudaMemcpyAsync(outBuffer, deviceBuffers[outputIndex], 1 * sizeof(float) * outputNumel, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        cudaFree(deviceBuffers[inputIndex]);
        cudaFree(deviceBuffers[outputIndex]);

        cudaStreamDestroy(stream);

        context->destroy();
    }
    engine->destroy();

    return 0;
}

