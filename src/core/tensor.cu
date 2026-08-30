#include "cudakernels/core/tensor.h"

namespace cudakernels {

// 显式实例化常用类型
template class Tensor<float>;

template class Tensor<double>;

template class Tensor<int>;

}  // namespace cudakernels
