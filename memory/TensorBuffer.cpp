//
// Created by baidu on 2017/6/5.
//

#include "TensorBuffer.h"

namespace nnet {
namespace memory {

Map<std::string, std::shared_ptr<TensorBuffer>> TensorBuffer::gTensorBuffers;
}
}