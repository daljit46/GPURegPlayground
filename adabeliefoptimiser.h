#pragma once

#include <vector>

// AdaBelief is an improved version of Adam that takes into account the curvature of the loss function.
// See here https://arxiv.org/abs/2010.07468
// This version is further enhanced by the idea in this paper https://arxiv.org/abs/2411.16085
// which consists in performing an element‐wise mask to the update such that only the components
// where the proposed update direction and the current gradient are aligned
// (i.e., have the same sign) are applied. This ensures that every step reliably reduces the loss
// and avoids potential overshooting or oscillations.
class AdaBeliefOptimiser {
public:
    struct Parameter {
        float value;
        float learning_rate;
    };

    AdaBeliefOptimiser(
        const std::vector<Parameter>& parameters,
        float beta1 = 0.7,
        float beta2 = 0.9999,
        float epsilon = 1e-8);

    const std::vector<Parameter>& step(const std::vector<float>& gradients);

private:
    std::vector<Parameter> m_parameters;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    int m_timeStep;
    std::vector<float> m_firstMoments;     // Exponential moving average of gradients (m_t)
    std::vector<float> m_secondMoments;    // Exponential moving average of squared deviations ((g_t - m_t)^2)
    std::vector<uint32_t> m_mask;
    std::vector<float> m_updates;
};

