/*
  Christopher Ginac
  
  image.cpp
*/

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "image.h"
#include <cmath>
#include <cuda_runtime.h>
using namespace std;
using namespace chrono;

const int numThreadsPerBlock = 16; // number of threads per block

__device__ bool inBounds(int row, int col, int N, int M)
/*checks to see if a pixel is within the image, returns true or false*/
{
	if (row >= N || row < 0 || col >= M || col < 0)
		return false;

	return true;
}

__global__ void rotatePixels(int* dst, int* src, int maxRows, int maxCols, float rads) {
	/*
	 * Given the block and the thread within it, calculate the row and column of the matrix.
	 */
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (c < maxCols && r < maxRows) {
		int r0 = maxRows / 2;
		int c0 = maxCols / 2;

		float sine, cosine;
		__sincosf(rads, &sine, &cosine);

		int r1 = (int)(r0 + ((r - r0) * cosine) - ((c - c0) * sine));
		int c1 = (int)(c0 + ((r - r0) * sine) + ((c - c0) * cosine));

		if (inBounds(r1, c1, maxRows, maxCols))
		{
			dst[r1 * maxCols + c1] = src[r * maxCols + c];
		}
	}
}

__global__ void enlargePixels(int* pixels, int* oldPixels, int oldMaxRows, int oldMaxCols, int maxCols, int value) {
	/*
	* Given the block and the thread within it, calculate the row and column of the matrix.
	*/
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < oldMaxCols && i < oldMaxRows) {
		int pixel = oldPixels[i * oldMaxCols + j];
		int enlargeRow = i * value;
		int enlargeCol = j * value;

		for (int c = enlargeRow; c < (enlargeRow + value); c++)
		{
			for (int d = enlargeCol; d < (enlargeCol + value); d++)
			{
				pixels[c * maxCols + d] = pixel;
			}
		}
	}
}


void checkCudaError(const cudaError_t &error, const char message[]) {
	if (error != cudaSuccess) {
		std::cerr << message << std::endl;
		std::cerr << cudaGetErrorString(error) << std::endl;
		exit(1);
	}
}

Image::Image()
/* Creates an Image 0x0 */
{
    N = 0;
    M = 0;
    Q = 0;
    
    pixelVal = NULL;
}

Image::Image(int numRows, int numCols, int grayLevels)
/* Creates an Image of numRows x numCols and creates the arrays for it*/
{    
    
    N = numRows;
    M = numCols;
    Q = grayLevels;
    
    pixelVal = new int [N * M];
}

Image::~Image()
/*destroy image*/
{
    N = 0;
    M = 0;
    Q = 0;
        
    delete pixelVal;
}

Image::Image(const Image& oldImage)
/*copies oldImage into new Image object*/
{    
    N = oldImage.N;
    M = oldImage.M;
    Q = oldImage.Q;
    
    pixelVal = new int [N * M];

    for(int i = 0; i < N * M; i++)
    {
		pixelVal[i] = oldImage.pixelVal[i];
    }
}

void Image::operator=(const Image& oldImage)
/*copies oldImage into whatever you = it to*/
{
    N = oldImage.N;
    M = oldImage.M;
    Q = oldImage.Q;
    
    pixelVal = new int [N * M];

	for (int i = 0; i < N * M; i++)
	{
		pixelVal[i] = oldImage.pixelVal[i];
	}
}

void Image::setImageInfo(int numRows, int numCols, int maxVal)
/*sets the number of rows, columns and graylevels*/
{
    N = numRows;
    M = numCols;
    Q = maxVal;
}

void Image::getImageInfo(int &numRows, int &numCols, int &maxVal)
/*returns the number of rows, columns and gray levels*/
{
    numRows = N;
    numCols = M;
    maxVal = Q;
}

int Image::getPixelVal(int row, int col)
/*returns the gray value of a specific pixel*/
{
    return pixelVal[row * M + col];
}


void Image::setPixelVal(int row, int col, int value)
/*sets the gray value of a specific pixel*/
{
    pixelVal[row * M + col] = value;
}

bool Image::inBounds(int row, int col)
/*checks to see if a pixel is within the image, returns true or false*/
{
    if(row >= N || row < 0 || col >=M || col < 0)
        return false;
    //else
    return true;
}

void Image::getSubImage(int upperLeftRow, int upperLeftCol, int lowerRightRow, 
    int lowerRightCol, Image& oldImage)
/*Pulls a sub image out of oldImage based on users values, and then stores it
    in oldImage*/
{
    int width, height;
    
    width = lowerRightCol - upperLeftCol;
    height = lowerRightRow - upperLeftRow;
    
    Image tempImage(height, width, Q);
    
    for(int i = upperLeftRow; i < lowerRightRow; i++)
    {
        for(int j = upperLeftCol; j < lowerRightCol; j++)
            tempImage.pixelVal[(i - upperLeftRow) * M + (j - upperLeftCol)] = oldImage.pixelVal[i * M + j];
    }
    
    oldImage = tempImage;
}

int Image::meanGray()
/*returns the mean gray levels of the Image*/
{
    int totalGray = 0;
    
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j++)
            totalGray += pixelVal[i * M + j];
    }
    
    int cells = M * N;
    
    return (totalGray / cells);
}

/* Enlarges Image and stores it in tempImage, resizes oldImage and stores the
 * larger image in oldImage
 */
void Image::enlargeImage(int value, Image& oldImage)

{
     int rows = oldImage.N * value, cols = oldImage.M * value;
     Image tempImage(rows, cols, oldImage.Q);

	 // Allocate d_pixels and d_oldPixels on the device.
	 int *d_pixels = nullptr, *d_oldPixels = nullptr;
	 checkCudaError(cudaMalloc((void**)&d_pixels, tempImage.N * tempImage.M * sizeof(int)),
		 "d_pixels - Allocation Error\n");
	 checkCudaError(cudaMalloc((void**)&d_oldPixels, oldImage.N * oldImage.M * sizeof(int)),
		 "d_oldPixels - Allocation Error\n");

	 // Copy from  oldImage.pixelVal to d_oldPixels.
	 checkCudaError(cudaMemcpy(d_oldPixels, oldImage.pixelVal, oldImage.N * oldImage.M * sizeof(int), cudaMemcpyHostToDevice),
		 "oldImage.pixelVal to d_oldPixels - Copy Error\n");

	 // Launch the enlargePixels kernel.
	 dim3 dGrid((oldImage.M + numThreadsPerBlock - 1) / numThreadsPerBlock, (oldImage.N + numThreadsPerBlock - 1) / numThreadsPerBlock, 1);
	 dim3 dBlock(numThreadsPerBlock, numThreadsPerBlock, 1);
	 enlargePixels << <dGrid, dBlock >> >(d_pixels, d_oldPixels, oldImage.N, oldImage.M, tempImage.M, value);
	 cudaDeviceSynchronize();
	 checkCudaError(cudaGetLastError(), "enlargePixels - Kernel Launch Error\n");

	 // Copy d_pixels to tempImage.pixelVal.
	 checkCudaError(cudaMemcpy(tempImage.pixelVal, d_pixels, tempImage.N * tempImage.M * sizeof(int), cudaMemcpyDeviceToHost),
		 "d_pixels to tempImage.pixelVal - Copy Error\n");

	 // Deallocate device memory.
	 cudaFree(d_pixels);
	 cudaFree(d_oldPixels);
     
     oldImage = tempImage;
}

void Image::shrinkImage(int value, Image& oldImage)
/*Shrinks image as storing it in tempImage, resizes oldImage, and stores it in
    oldImage*/
{
    int rows, cols, gray;
    
    rows = oldImage.N / value;
    cols = oldImage.M / value;  
    gray = oldImage.Q; 
    
    Image tempImage(rows, cols, gray);
    
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
            tempImage.pixelVal[i * M + j] = oldImage.pixelVal[(i * value) * M + (j * value)];
    }    
    oldImage = tempImage;
}

void Image::reflectImage(bool flag, Image& oldImage)
/*Reflects the Image based on users input*/
{
    int rows = oldImage.N;
    int cols = oldImage.M;
    Image tempImage(oldImage);
    if(flag == true) //horizontal reflection
    {
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                tempImage.pixelVal[(rows - (i + 1)) * M + j] = oldImage.pixelVal[i * M + j];
        }
    }
    else //vertical reflection
    {
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                tempImage.pixelVal[i * M + (cols - (j + 1))] = oldImage.pixelVal[i * M + j];
        }
    } 
    
    oldImage = tempImage;
}
        
void Image::translateImage(int value, Image& oldImage)
/*translates image down and right based on user value*/
{
    int rows = oldImage.N;
    int cols = oldImage.M;
    int gray = oldImage.Q;
    Image tempImage(N, M, Q);
    
    for(int i = 0; i < (rows - value); i++)
    {
        for(int j = 0; j < (cols - value); j++)
            tempImage.pixelVal[(i + value) * M + (j + value)] = oldImage.pixelVal[i * M + j];
    }
    
    oldImage = tempImage;
}

// Based on users input and rotates it around the center of the image
void Image::rotateImage(int theta, Image& oldImage)
{
	steady_clock::time_point ts, te;
    int rows = oldImage.N, cols = oldImage.M;
	float rads = (theta * 3.14159265) / 180.0;
    Image tempImage(rows, cols, oldImage.Q);

	// Allocate d_pixels and d_oldPixels on the device.
	ts = steady_clock::now();
	int *d_pixels = nullptr, *d_oldPixels = nullptr;
	checkCudaError(cudaMalloc((void**)&d_pixels, tempImage.N * tempImage.M * sizeof(int)),
		"d_pixels - Allocation Error\n");
	checkCudaError(cudaMalloc((void**)&d_oldPixels, oldImage.N * oldImage.M * sizeof(int)),
		"d_oldPixels - Allocation Error\n");
	te = steady_clock::now();
	auto ms = duration_cast<milliseconds>(te - ts);
	std::cout << "cudaMalloc took " << ms.count() << " millisecs" << std::endl;

	// Copy from oldImage.pixelVal to d_oldPixels.
	ts = steady_clock::now();
	checkCudaError(cudaMemcpy(d_oldPixels, oldImage.pixelVal, oldImage.N * oldImage.M * sizeof(int), cudaMemcpyHostToDevice),
		"oldImage.pixelVal to d_oldPixels - Copy Error\n");
	te = steady_clock::now();
	ms = duration_cast<milliseconds>(te - ts);
	std::cout << "cudaMemcpy took " << ms.count() << " millisecs" << std::endl;

	// Launch the rotatePixels kernel.
	ts = steady_clock::now();
	dim3 dGrid((cols + numThreadsPerBlock - 1) / numThreadsPerBlock, (rows + numThreadsPerBlock - 1) / numThreadsPerBlock, 1);
	dim3 dBlock(numThreadsPerBlock, numThreadsPerBlock, 1);
	rotatePixels << <dGrid, dBlock >> >(d_pixels, d_oldPixels, rows, cols, rads);
	cudaDeviceSynchronize();
	checkCudaError(cudaGetLastError(), "rotatePixels - Kernel Launch Error\n");
	te = steady_clock::now();
	ms = duration_cast<milliseconds>(te - ts);
	std::cout << "rotate image kernel took " << ms.count() << " millisecs" << std::endl;

	// Copy d_pixels to tempImage.pixelVal.
	ts = steady_clock::now();
	checkCudaError(cudaMemcpy(tempImage.pixelVal, d_pixels, tempImage.N * tempImage.M * sizeof(int), cudaMemcpyDeviceToHost),
		"d_pixels to tempImage.pixelVal - Copy Error\n");
	te = steady_clock::now();
	ms = duration_cast<milliseconds>(te - ts);
	std::cout << "cudaMemcpy took " << ms.count() << " millisecs" << std::endl;

	// Deallocate device memory.
	ts = steady_clock::now();
	cudaFree(d_pixels);
	cudaFree(d_oldPixels);
	te = steady_clock::now();
	ms = duration_cast<milliseconds>(te - ts);
	std::cout << "cudaFree took " << ms.count() << " millisecs" << std::endl;
	
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(tempImage.pixelVal[i * M + j] == 0)
                tempImage.pixelVal[i * M + j] = tempImage.pixelVal[i * M + (j + 1)];
        }
    }

    oldImage = tempImage;
}
            
Image Image::operator+(const Image &oldImage)
 /*adds images together, half one image, half the other*/
{
    Image tempImage(oldImage);
    
    int rows, cols;
    rows = oldImage.N;
    cols = oldImage.M;
    
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
            tempImage.pixelVal[i * M + j] = (pixelVal[i * M + j] + oldImage.pixelVal[i * M + j]) / 2;
    }
    
    return tempImage;
}

Image Image::operator-(const Image& oldImage)
/*subtracts images from each other*/
{
    Image tempImage(oldImage);
    
    int rows, cols;
    rows = oldImage.N;
    cols = oldImage.M;
    int tempGray = 0;
    
     for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            
            tempGray = abs(pixelVal[i * M + j] - oldImage.pixelVal[i * M + j]);
            if(tempGray < 35)// accounts for sensor flux
                tempGray = 0;
            tempImage.pixelVal[i * M + j] = tempGray;
        }
    
    }
    
    return tempImage;    
}

void Image::negateImage(Image& oldImage)
/*negates image*/
{
    int rows, cols, gray;
    rows = N;
    cols = M;
    gray = Q;
    
    Image tempImage(N,M,Q);
    
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
            tempImage.pixelVal[i * M + j] = -(pixelVal[i * M + j]) + 255;
    }
    
    oldImage = tempImage;
}
