**Tên nhóm: HTH**

**Thành viên:**
- 20120494 - Lê Xuân Huy
- 20120089 - Lê Xuân Hoàng
- 20120422 - Nguyễn Thị Ánh Tuyết

Cách chạy: Xem demo [Run-Lenet5](https://colab.research.google.com/drive/1B3C4PLaVH6pxKBII19HIXLLOWmGbQFMr?usp=sharing)

```
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/local/cuda/bin/nvcc ..
make
```
Test:
- So sánh với cpu: chạy `./demo test <phiên bản>`.
- So sánh 2 phiên bản: chạy `./demo test <phiên bản 1> <phiên bản 2>`.
Run:
- Chạy `./demo run <phiên bản>`

