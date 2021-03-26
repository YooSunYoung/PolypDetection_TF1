vai_q_tensorflow quantize \
	--input_frozen_graph results/model.pb \
	--input_nodes input \
	--input_shapes ?,227,227,3 \
	--output_nodes  output \
	--input_fn input_fn.calib_input \
	--method 0 \
	--gpu 0 \
	--calib_iter 30 \
	--output_dir /workspace/quantize_results \
	--weight_bit 8 \
	--activation_bit 8 

