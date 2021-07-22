编译 apex ，需要 nvcc 和 pytorch 版本完全一致

export CUDA_HOME=/usr/local/cuda-10.1/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:$PATH
nvcc -V


python tools/demo.py image -n yolox-s -c ../yolox_s.pth.tar --path ../assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result

python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 32 -d 1 --conf 0.001

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.587
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.705


python tools/eval.py -n  yolox-tiny -c ../yolox_tiny.pth.tar -b 2 -d 1 --conf 0.001


python tools/train.py -n yolox-tiny -d 1 -b 2 -o


python tools/train.py -n yolox-s -d 1 -b 2 -o


