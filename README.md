# Gaussian Blur Filter from Scratch with CUDA

This repository demonstrates how to apply a **Gaussian Blur** filter to an image using the parallel processing capabilities of **CUDA**.

---

## 🚀 Overview

Gaussian Blur is a widely used filter in image processing for smoothing and noise reduction. This project implements the filter from scratch using CUDA to exploit GPU parallelism and accelerate processing.

We utilize **shared memory** in CUDA to efficiently access neighboring pixels and reduce global memory latency, which is especially important when performing convolution operations like Gaussian filtering.

---

## 🛠️ How Does Gaussian Blur Work in CUDA?

We use a 5x5 Gaussian kernel to apply the blur. Each CUDA thread is responsible for computing the blurred value of a single pixel using its local neighborhood. To avoid redundant global memory access, threads cooperatively load a tile of image pixels into shared memory.

```cpp
__shared__ unsigned char tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];
```

This creates a shared memory buffer (a tile) that includes padding (+2 in each direction) to store the necessary 5x5 neighborhood around each pixel.

---

## 📌 Key CUDA Variables

```cpp
int tx = threadIdx.x; // Local X coordinate within the block
int ty = threadIdx.y; // Local Y coordinate within the block

int x = blockIdx.x * BLOCK_SIZE + tx; // Global X coordinate in the image
int y = blockIdx.y * BLOCK_SIZE + ty; // Global Y coordinate in the image

int shared_x = tx + 2; // Shared memory X index (offset by padding)
int shared_y = ty + 2; // Shared memory Y index (offset by padding)
```

Each thread computes its **global image coordinates** and maps them to corresponding **shared memory locations**.

---

## 🧠 Loading into Shared Memory

```cpp
if (x < width && y < height)
    tile[shared_y][shared_x] = input[y * width + x]; // Convert 2D to 1D index
```

Each thread loads the image pixel at its global coordinates into shared memory. Additional threads load neighboring pixels to cover the full 5x5 window required for Gaussian blur.

Using shared memory greatly reduces the latency of reading pixel neighborhoods compared to accessing global memory multiple times.

---

## 💻 Requirements

- CUDA-enabled GPU
- NVIDIA CUDA Toolkit
- OpenCV (for image loading and saving)

---

## 📷 Sample Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/kayraobi/CUDA-GaussianBlur-implementation
   cd cuda-gaussian-blur
   ```

2. Compile:
   ```bash
   nvcc -o gaussian main.cu `pkg-config --cflags --libs opencv4`
   ```

3. Run:
 ./gaussian

---

## 📊 Performance

By using shared memory, this implementation significantly outperforms a naive global-memory-only version, especially on large images.

---

## 📄 License

MIT License. Feel free to use, modify, and share.

---

## 🤝 Contributions

Pull requests and suggestions are welcome. If you find a bug or have an idea to improve performance or structure, feel free to open an issue.
