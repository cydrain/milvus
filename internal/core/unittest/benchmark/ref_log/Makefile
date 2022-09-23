###################################################################################################
# Test Segcore index
test_segcore_all: test_segcore_idmap test_segcore_ivf_flat test_segcore_ivf_sq8 test_segcore_ivf_pq test_segcore_hnsw

test_segcore_idmap:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_IDMAP" | tee test_segcore_idmap.log
test_segcore_ivf_flat:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_IVF_FLAT" | tee test_segcore_ivf_flat.log
test_segcore_ivf_sq8:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_IVF_SQ8" | tee test_segcore_ivf_sq8.log
test_segcore_ivf_pq:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_IVF_PQ" | tee test_segcore_ivf_pq.log
test_segcore_ivf_hnsw:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_IVF_HNSW" | tee test_segcore_ivf_hnsw.log
test_segcore_hnsw:
	./benchmark_segcore_float --gtest_filter="Benchmark_segcore_float.TEST_HNSW" | tee test_segcore_hnsw.log

###################################################################################################
# Test Segcore index range
test_segcore_range_all: test_segcore_range_idmap test_segcore_range_ivf_flat_nm test_segcore_range_ivf_sq8 test_segcore_range_ivf_pq test_segcore_range_hnsw

test_segcore_range_idmap:
	./benchmark_segcore_float_range --gtest_filter="Benchmark_segcore_float_range.TEST_IDMAP" | tee test_segcore_range_idmap.log
test_segcore_range_ivf_flat_nm:
	./benchmark_segcore_float_range --gtest_filter="Benchmark_segcore_float_range.TEST_IVF_FLAT" | tee test_segcore_range_ivf_flat.log
test_segcore_range_ivf_sq8:
	./benchmark_segcore_float_range --gtest_filter="Benchmark_segcore_float_range.TEST_IVF_SQ8" | tee test_segcore_range_ivf_sq8.log
test_segcore_range_ivf_pq:
	./benchmark_segcore_float_range --gtest_filter="Benchmark_segcore_float_range.TEST_IVF_PQ" | tee test_segcore_range_ivf_pq.log
test_segcore_range_hnsw:
	./benchmark_segcore_float_range --gtest_filter="Benchmark_segcore_float_range.TEST_HNSW" | tee test_segcore_range_hnsw.log

###################################################################################################
# Test Segcore binary index
test_segcore_binary_all: test_segcore_binary_idmap test_segcore_binary_ivf_flat

test_segcore_binary_idmap:
	./benchmark_segcore_binary --gtest_filter="Benchmark_segcore_binary.TEST_BINARY_IDMAP" | tee test_segcore_binary_idmap.log
test_segcore_binary_ivf_flat:
	./benchmark_segcore_binary --gtest_filter="Benchmark_segcore_binary.TEST_BINARY_IVF_FLAT" | tee test_segcore_binary_ivf_flat.log

###################################################################################################
# Test Segcore binary index range
test_segcore_binary_range_all: test_segcore_binary_range_idmap test_segcore_binary_range_ivf_flat

test_segcore_binary_range_idmap:
	./benchmark_segcore_binary_range --gtest_filter="Benchmark_segcore_binary_range.TEST_BINARY_IDMAP" | tee test_segcore_binary_range_idmap.log
test_segcore_binary_range_ivf_flat:
	./benchmark_segcore_binary_range --gtest_filter="Benchmark_segcore_binary_range.TEST_BINARY_IVF_FLAT" | tee test_segcore_binary_range_ivf_flat.log

