docker run -v ${PWD}:/workspace -w /workspace -d -it --rm --name object_detection xilinx/vitis-ai:latest
docker exec object_detection /opt/vitis_ai/conda/bin/conda env list
docker exec object_detection /opt/vitis_ai/conda/bin/conda init bash
docker exec object_detection /opt/vitis_ai/conda/bin/conda activate vitis-ai-tensorflow
docker stop object_detection