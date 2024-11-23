#pragma once

#include <vector>

class AdamOptimizer {
public:
    struct Parameter {
        float value;
        float learning_rate;
    };
    AdamOptimizer(
        const std::vector<Parameter>& parameters,
        float beta1 = 0.8, // higer beta1 means more weight on past gradients
        float beta2 = 0.9999, // higher beta2 means more stable updates
        float epsilon = 1e-8);

    const std::vector<Parameter> &step(const std::vector<float>& gradients);

private:
    std::vector<Parameter> m_parameters;
    float m_beta1; // First moment decay rate
    float m_beta2; // Second moment decay rate
    float m_epsilon;
    int m_timeStep;
    std::vector<float> m_firstMoments;
    std::vector<float> m_secondMoments;
};
