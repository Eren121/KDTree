#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <stack>
#include <ostream>
#include <thrust/device_vector.h>
#include "Math.h"


struct KDTreeBase
{
    /**
     * @brief Represent each node of the k-d tree.
     *
     * @note Should be public to allow thrust::device_vector<Node> but should not be used outside the class.
     */
    struct Node
    {
        /**
         * @brief Use compact representation to save memory.
         *
         * We consider zero-initialization should be the empty node (because its simpler to treat).
         * So the fields should be set accordingly.
         * It can be nasty bugs if the values are not initialized.
         */
        struct Header
        {
            // All bitfields should have the same type to be on the same byte,
            // and they are aligned by their underlying type,
            // so if we declare them "unsigned char" Header will be 4 bytes, wasting 3.

            unsigned char hasChildren : 1; ///< 0 if the node is a leaf, 1 otherwise (zero-initialization is empty node)
            unsigned char dim : 2; ///< Dimension of the split, in [0;3)
        };

        // Check optimal size
        static_assert(sizeof(Header) == 1);

        Header header{}; ///< Header, zero-initialized by default.
        float p; ///< Line of the split

        /**
         * @return The split line. We don't store it directly to permit a compact representation.
         */
        CUDA_BOTH Line line() const
        {
            Line ret;
            ret.dim = header.dim;
            ret.p = p;

            return ret;
        }

        Triangle::ID* mesh;
        int meshSize;

        /**
         * @return true if this node is a leaf node (it has no children).
         */
        CUDA_BOTH bool leaf() const
        {
            // left or right it doesn't matter
            return !header.hasChildren;
        }
    };

    using NodeID = int;

    /**
     * @brief Compute the bounding box of a set of triangles.
     * .
     * @param mesh
     *      The triangles to make the bounding box from.
     *      If the set of points is empty, the returned value is zero.
     * @return
     *      The bounding box of theses triangles..
     *      The bounding box will be always of minimum size.
     */
    static AABB computeBoundingBox(const Mesh &mesh);

    /**
     * Find the closest point on a triangle from a given point.
     */
    CUDA_BOTH static Point findClosestPointOnTriangle(const Point& query, const Triangle& triangle);

    /**
     * Check if a 3D point is inside a triangle
     */

    CUDA_BOTH static void store_min(float& current, float newValue)
    {
        if(current > newValue) {
            current = newValue;
        }
    }

    CUDA_BOTH static void store_max(float& current, float newValue)
    {
        if(current < newValue) {
            current = newValue;
        }
    }

    struct Heuristics
    {
        enum Dim {
            DIM_3D, // Rotate x, y, z, x, y, z...
            DIM_2D // Rotate x, y, x, y...
        };

        Dim dim = DIM_3D;

        int maxNodeSize = 100;
        int maxLevel = 5;
    };
};

/**
 * @brief K-d tree of 3D triangles.
 *
 * @see Variable names, and algorithm inspired by https://youtu.be/TrqK-atFfWY?t=2567
 */
class KDTree : public KDTreeBase
{
public:
    KDTree() = default;
    explicit KDTree(const Mesh& mesh, Heuristics heuristics = {});

    [[nodiscard]] const auto& getMesh() const { return m_mesh; }
    [[nodiscard]] const auto& getBounds() const { return m_rootAABB; }

    /**
     * @return
     *      The maximum number of nodes.
     *      It depend on the max level value.
     *      The number of non-empty nodes stored is probably way less.
     */
    [[nodiscard]] int getMaxNodesCount() const
    {
        // Maximum number of node in a perfect binary tree where the deepest level is m_maxLevel (root is at level 0)
        return (1 << (m_heuristics.maxLevel + 1)) - 1;
    }

    template<typename T>
    T& summarize(T& out)
    {
        out << "KDTree summary:" << std::endl;
        out << "    Max level: " << m_heuristics.maxLevel << std::endl;
        out << "    Max nodes: " << getMaxNodesCount() << std::endl;
        out << "    Non-null nodes: " << (m_totalLeafNodes * 2 - 1) << std::endl; // Handshaking Lemma
        out << "    Leaves: " << m_totalLeafNodes << std::endl;
        out << "    Triangles in leaves: " << m_leavesBuffer.size()
            << " (" << m_leavesBuffer.size() * sizeof(Triangle::ID) << " bytes)" << std::endl;

        return out;
    }

private:
    void init();

private:
    /**
     * Buffer to store the triangle points.
     * Each triangle m_mesh[i] unique index is i.
     */
    thrust::device_vector<Triangle> m_mesh;

    /**
     * Buffer to store the nodes.
     * Store a binary tree as an array representation.
     * The array can be very sparse because there are a lot of null nodes.
     * However, the maximum count of leaves nodes is relatively small (e.g. 2^15) so it is not costly.
     * We don't need to flag the null node because they are never accessed, all of their parent nodes are leaves.
     */
    thrust::device_vector<Node> m_nodes;

    /**
     * Buffer to store each node content.
     * Each node has an contiguous part of the array.
     * The contiguous part may not overlap with other contiguous parts (i.e. each node has its unique part).
     * So there are no race conditions on its access.
     */
    thrust::device_vector<Triangle::ID> m_leavesBuffer;

    AABB m_rootAABB;
    int m_totalLeafNodes{0}; ///< Total count of leaves
    Heuristics m_heuristics;

public:
    friend class KDTreeDevicePtr;
};