#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a tensor
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Hello World!! Here are my torch tensors::" << tensor << "!?" << std::endl;
    return 0;
    
}

