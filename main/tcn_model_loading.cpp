#include <torch/torch.h>
#include <torch/script.h> // For loading ScriptModule
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono> // For measuring execution time

// Load data from a text file into a vector of floats
std::vector<std::vector<float>> loadSimData(const std::string& file_path, int rows, int cols) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> data[i][j];
        }
    }
    file.close();
    return data;
}

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Parameters
    int start_size = 0;
    int validation_size = 3000;
    int data_set_size = start_size + validation_size;

    // Load simulation data
    std::string path1 = "sim_data/dataset/x_data_all_scara_AKF_15000.txt";
    int rows = data_set_size;  // Replace with actual row size
    int cols = 10;            // Replace with actual column size
    auto x_input_data_all = loadSimData(path1, rows, cols);

    // Load pre-trained model
    std::string model_path = "sim_data/model/x_tcn_model_fea10_ker10_num[64]_epo500.pt";
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Start inference
    std::vector<std::vector<float>> x_tcn_output_data;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int k = start_size; k < data_set_size; ++k) {
        // Prepare input tensor
        auto x_input_data = torch::from_blob(x_input_data_all[k].data(), {1, 1, cols}, torch::kFloat32).to(device);

        // Perform inference
        torch::Tensor output = model.forward({x_input_data}).toTensor();
        std::vector<float> output_vec(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
        x_tcn_output_data.push_back(output_vec);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Inference completed in: " << elapsed_time.count() << " seconds." << std::endl;

    // Save output data to file
    std::ofstream output_file("sim_data/result/x_tcn_output_data_sim.txt");
    for (const auto& row : x_tcn_output_data) {
        for (size_t i = 0; i < row.size(); ++i) {
            output_file << row[i];
            if (i < row.size() - 1) output_file << " ";
        }
        output_file << "\n";
    }
    output_file.close();

    std::cout << "Results saved to sim_data/result/x_tcn_output_data_sim.txt" << std::endl;
    return 0;
}