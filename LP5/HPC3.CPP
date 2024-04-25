#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

struct ReductionResults {
    int min_val;
    int max_val;
    int sum_val;
    double average_val;
};

ReductionResults parallel_reduction(const std::vector<int>& data) {
    ReductionResults results;
    results.min_val = data[0];
    results.max_val = data[0];
    results.sum_val = 0;
    
    #pragma omp parallel for reduction(min:results.min_val) reduction(max:results.max_val) reduction(+:results.sum_val)
    for (size_t i = 0; i < data.size(); ++i) {
        results.min_val = std::min(results.min_val, data[i]);
        results.max_val = std::max(results.max_val, data[i]);
        results.sum_val += data[i];
    }
    
    results.average_val = static_cast<double>(results.sum_val) / data.size();
    
    return results;
}

int main() {
    std::vector<int> data = {5, 10, 15, 20, 25, 30, 35, 40};
    ReductionResults results = parallel_reduction(data);
    
    std::cout << "Minimum: " << results.min_val << std::endl;
    std::cout << "Maximum: " << results.max_val << std::endl;
    std::cout << "Sum: " << results.sum_val << std::endl;
    std::cout << "Average: " << results.average_val << std::endl;
    
    return 0;
}
