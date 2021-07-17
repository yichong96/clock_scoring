//
//  InferenceModule.mm
//  clockscoring
//
//  Created by Yi Chong Ong on 6/7/21.
//

#import "InferenceModule.h"
#import <LibTorch.h>

const int input_width = 320;
const int input_height = 320;
const int threshold = 0.5;


@implementation InferenceModule {
    @protected torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)detectImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, { 3, input_width, input_height }, at::kFloat);
        tensor = tensor.clone();
//        at::Tensor tensor = torch::from_blob(imageBuffer, { input_width, input_height, 3 }, at::kFloat);
//        tensor = tensor.permute({2, 0, 1});
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
//        torch::jit::script::Module container = torch::jit::load("/Users/ongyichong/ComputerScience/clock_scoring/container.pt");
//        at::Tensor tensor = container.attr("sample_clock").toTensor();
        std::vector<torch::Tensor> v;
        v.push_back(tensor);
        auto outputTuple = _impl.forward({at::TensorList(v)}).toTuple();
        auto outputDict = outputTuple->elements()[1].toList().get(0).toGenericDict();
        //std::cout << outputDict;
        auto boxesTensor = outputDict.at("boxes").toTensor();
        auto scoresTensor = outputDict.at("scores").toTensor();
        auto labelsTensor = outputDict.at("labels").toTensor();

        float* boxesBuffer = boxesTensor.data_ptr<float>();
        if (!boxesBuffer) {
            return nil;
        }
        float* scoresBuffer = scoresTensor.data_ptr<float>();
        if (!scoresBuffer) {
            return nil;
        }
        int64_t* labelsBuffer = labelsTensor.data_ptr<int64_t>();
        if (!labelsBuffer) {
            return nil;
        }

        NSMutableArray* results = [[NSMutableArray alloc] init];
        long num = scoresTensor.numel();
        for (int i = 0; i < num; i++) {
            if (scoresBuffer[i] < threshold)
                continue;

            [results addObject:@(boxesBuffer[4 * i])];
            [results addObject:@(boxesBuffer[4 * i + 1])];
            [results addObject:@(boxesBuffer[4 * i + 2])];
            [results addObject:@(boxesBuffer[4 * i + 3])];
            [results addObject:@(scoresBuffer[i])];
            [results addObject:@(labelsBuffer[i])];
        }
        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
