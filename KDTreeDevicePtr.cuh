#pragma once

#include "KDTree.cuh"

/**
 * Contains pointers to device memory.
 * Can be copied to CUDA kernel to access the KDTree.
 */
class KDTreeDevicePtr : KDTreeBase
{
public:
    /**
     * Can be copied from a KDTree.
     * It will copy pointers to the device vectors.
     */
    explicit KDTreeDevicePtr(const KDTree& tree);

    __device__ void searchRecursive(const Point& pos, float& currentDist, Triangle::ID& currentID,
                         Point& currentPoint) const;

    /**
     * Compute the nearest point on the mesh from the query position.
     *
     * @param pos The query position.
     *
     * @return A pair (point, id) where id is the ID of the triangle.
     */
    __device__ NPQueryRet findNearestPointOnMesh(const Point& pos) const;

    /**
     * View on a portion of memory, non-owning array.
     * Like `std::span` class on device.
     * There are no `thrust::span` class for now.
     *
     * @tparam The type to store. May be const to denote a constant view.
     */
    template<typename T>
    struct Span
    {
        /**
         * @tparam ThrustVector To permit const and non-const vectors.
         */
        template<typename ThrustVector>
        explicit Span(ThrustVector& v)
            : data(thrust::raw_pointer_cast(v.data())),
              size(v.size())
        {
        }

        __device__ T& operator[](size_t i) { return data[i]; }
        __device__ const T& operator[](size_t i) const { return data[i]; }

        T* data;
        size_t size;
    };

private:
    Span<const Triangle> m_mesh;
    Span<const Node> m_nodes;
    Span<const Triangle::ID> m_leavesBuffer;
    AABB m_rootAABB;
    int m_maxLevel;
    int m_totalLeafNodes;
};