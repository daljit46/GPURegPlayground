#include "adabeliefoptimiser.h"
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

AdaBeliefOptimiser::AdaBeliefOptimiser(
    const std::vector<Parameter>& parameters,
    float beta1,
    float beta2,
    float epsilon)
    : m_parameters(parameters),
    m_beta1(beta1),
    m_beta2(beta2),
    m_epsilon(epsilon),
    m_timeStep(0),
    m_firstMoments(parameters.size(), 0.0F),
    m_secondMoments(parameters.size(), 0.0F),
    m_mask(parameters.size(), 0),
    m_updates(parameters.size(), 0.0F)
{
}

const std::vector<AdaBeliefOptimiser::Parameter>&
AdaBeliefOptimiser::step(const std::vector<float>& gradients)
{
    ++m_timeStep;

    // First pass: compute moment estimates, the update direction, and binary masks
    for (size_t i = 0; i < m_parameters.size(); ++i) {
        m_firstMoments[i] = m_beta1 * m_firstMoments[i] + (1.0F - m_beta1) * gradients[i];
        float diff = gradients[i] - m_firstMoments[i];
        m_secondMoments[i] = m_beta2 * m_secondMoments[i] + (1.0F - m_beta2) * diff * diff;
        const float firstMomentCorrected = m_firstMoments[i] / (1.0F - std::pow(m_beta1, m_timeStep));
        const float secondMomentCorrected = m_secondMoments[i] / (1.0F - std::pow(m_beta2, m_timeStep));
        m_updates[i] = firstMomentCorrected / (std::sqrt(secondMomentCorrected) + m_epsilon);
        // Create mask: update only if update and gradient are aligned (i.e. product > 0)
        m_mask[i] = (m_updates[i] * gradients[i] > 0.0F) ? 1 : 0;
    }

    // Compute the average mask value across all parameters
    const float maskSum = std::accumulate(m_mask.begin(), m_mask.end(), 0);
    const float maskMean = maskSum / static_cast<float>(m_mask.size());

    // Second pass: apply cautious update to each parameter
    for (size_t i = 0; i < m_parameters.size(); ++i) {
        // If the mask is 0, no update is applied; otherwise, scale the update to compensate for the overall sparsity.
        m_parameters[i].value -= m_parameters[i].learning_rate * (m_updates[i] * m_mask[i] / (maskMean + m_epsilon));
    }

    return m_parameters;
}

