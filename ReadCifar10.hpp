#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>


void ReadFiles(std::string fileName, std::vector<std::vector<float>>& Data, std::vector<unsigned>& Data_labels)
{
	std::ifstream binaryFile(fileName, std::ios::binary | std::ios::in);
	unsigned imageSize = 3073;
	std::unique_ptr<unsigned char[]> imagePtr(new unsigned char[imageSize]);

	if (binaryFile.is_open())
	{
		while (binaryFile.read(reinterpret_cast<char*>(imagePtr.get()), imageSize))
		{
			// assigning labels
			Data_labels.push_back(imagePtr.get()[0]);


			// assigning data
			std::vector<float> tempData(3072);
			/*unsigned elem = 1;
			for (unsigned c = 0; c != 3; c++) {
			for (unsigned row = 0; row != 32; row++) {
			for (unsigned col = 0; col != 32; col++) {
			tempData[c][row][col] = float(imagePtr.get()[elem++]) / 255.0f - 0.5f;
			}
			}
			}*/
			for (unsigned elem = 0; elem != tempData.size(); elem++)
			{
				tempData[elem] = float(imagePtr.get()[elem + 1]) / 255.0f - 0.5f;
			}
			Data.push_back(tempData);
		}
	}
	else {
		std::cout << "Error opening the file... " << std::endl;
		std::cin.get();
	}
}

void CreateTrainData(std::vector<std::vector<float>>& train_data, std::vector<unsigned>& train_data_labels)
{
	ReadFiles("cifar/data_batch_1.bin", train_data, train_data_labels);
	ReadFiles("cifar/data_batch_2.bin", train_data, train_data_labels);
	ReadFiles("cifar/data_batch_3.bin", train_data, train_data_labels);
	ReadFiles("cifar/data_batch_4.bin", train_data, train_data_labels);
}

void CreateTestData(std::vector<std::vector<float>>& test_data, std::vector<unsigned>& test_data_labels)
{
	ReadFiles("cifar/test_batch.bin", test_data, test_data_labels);
}