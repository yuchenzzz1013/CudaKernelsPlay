# 与 src/ops 完全对称的测试目录结构
set(CUDAKERNELS_TESTS
  ops/blas/test_add.cu
  ops/blas/test_gemv.cu
  ops/blas/test_sgemm.cu
  ops/norm/test_rms_norm.cu
  ops/activation/test_swiglu.cu
  ops/reduce/test_sum_reduce.cu
  ops/reduce/test_max_reduce.cu
  ops/transform/test_transpose.cu
  ops/transform/test_softmax.cu
  ops/transform/test_topk.cu
  ops/attention/test_mha.cu
)

foreach(test_file IN LISTS CUDAKERNELS_TESTS)
  get_filename_component(test_name ${test_file} NAME_WE)

  add_executable(${test_name} ${test_file})

  target_link_libraries(${test_name} PRIVATE cudakernels)
endforeach()
