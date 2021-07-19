#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class repeat_vector_layer : public layer
{
public:
    explicit repeat_vector_layer(const std::string& name,
        std::size_t n)
        : layer(name),
        n_(n)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        tensors result;
        for (auto i = 0; i < n_; ++i)
            result.push_back(input);
        return result;
    }
    size_t n_;
};

} } // namespace fdeep, namespace internal
