#pragma once

#include <rerun.hpp>

#include <Eigen/Core>
#include <cassert>

template <>
struct rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>> {
    /// Borrow for non-temporary.
    Collection<rerun::Position3D> operator()(const std::vector<Eigen::Vector3f>& container) {
        return Collection<rerun::Position3D>::borrow(container.data(), container.size());
    }

    // Do a full copy for temporaries (otherwise the data might be deleted when the temporary is destroyed).
    Collection<rerun::Position3D> operator()(std::vector<Eigen::Vector3f>&& container) {
        std::vector<rerun::Position3D> positions(container.size());
        memcpy(positions.data(), container.data(), container.size() * sizeof(Eigen::Vector3f));
        return Collection<rerun::Position3D>::take_ownership(std::move(positions));
    }
};