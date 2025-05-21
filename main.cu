#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

__global__ void gaussian_filter(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}};

    int sum = 0;
    int weight = 0;

    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                int neighbor_idx = ny * width + nx;
                int kernel_value = kernel[dy + 2][dx + 2];
                sum += input[neighbor_idx] * kernel_value;
                weight += kernel_value;
            }
        }
    }

    int idx = y * width + x;
    output[idx] = sum / weight;
}

__global__ void vector_addition(const unsigned char *a, const unsigned char *b, unsigned char *out, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        out[id] = min((int)a[id] + (int)b[id], 255);  // saturate to avoid overflow
}


int main()
{
    cv::Mat image = cv::imread("Tr-gl_0010.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "image couldn't load\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int size = width * height;

    unsigned char *image_1d = new unsigned char[size];
    unsigned char *output_image = new unsigned char[size];

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            image_1d[y * width + x] = image.at<uchar>(y, x);

    unsigned char *d_input, *d_output, *d_result;
    cudaMalloc(&d_input, size * sizeof(unsigned char));
    cudaMalloc(&d_output, size * sizeof(unsigned char));
    cudaMalloc(&d_result, size * sizeof(unsigned char)); 

    cudaMemcpy(d_input, image_1d, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    gaussian_filter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    int blockSize1D = 256;
    int gridSize1D = (size + blockSize1D - 1) / blockSize1D;
    vector_addition<<<gridSize1D, blockSize1D>>>(d_input, d_output, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(output_image, d_result, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat output_mat(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            output_mat.at<uchar>(y, x) = output_image[y * width + x];

    cv::Mat diff_raw, diff;
    cv::absdiff(image, output_mat, diff_raw);
    cv::convertScaleAbs(diff_raw, diff, 2.0);

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{image, output_mat, diff}, combined);

    cv::imshow("Original | Blurred+Original | Difference (x5 contrast)", combined);
    cv::imwrite("blurred_output.jpg", output_mat);
    cv::imwrite("difference_output.jpg", diff);
    cv::imwrite("combined_output.jpg", combined);
    cv::waitKey(0);

    delete[] image_1d;
    delete[] output_image;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);

    return 0;
}
