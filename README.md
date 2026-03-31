pip deps use CUDA wheel not CPU

python3.10 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
cargo run

Keep running cargo run its a race condition

That should end with:

malloc(): largebin double linked list corrupted (bk)







C
----------------------------------
gcc -o c_main c_main.c -ldl

LD_LIBRARY_PATH=.venv/lib/python3.10/site-packages/nvidia/cublas/lib:.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:.venv/lib/python3.10/site-packages/nvidia/cufft/lib:.venv/lib/python3.10/site-packages/nvidia/curand/lib:.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib ./c_main
