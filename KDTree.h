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

/**
 * @brief K-d tree of 3D triangles.
 *
 * @see Variable names, and algorithm inspired by https://youtu.be/TrqK-atFfWY?t=2567
 */
class KDTree
{
public:
    struct Point
    {
        float x, y, z;

        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wreturn-type"

        float& operator[](int i)
        {
            assert(i < 3 && i >= 0);

            switch(i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
            }

            assert(false && "Invalid index");
        }

        #pragma GCC diagnostic pop

        const float& operator[](int i) const
        {
            return const_cast<Point&>(*this)[i];
        }

        [[nodiscard]] float distance(const Point& other) const
        {
            return sqrtf(distanceSquared(other));
        }

        [[nodiscard]] float distanceSquared(const Point& other) const
        {
            const float dx = (x - other.x);
            const float dy = (y - other.y);
            const float dz = (z - other.z);
            return dx*dx + dy*dy + dz*dz;
        }

        Point operator-(const Point& other) const
        {
            return {x - other.x, y - other.y, z - other.z};
        }

        Point operator*(float f) const
        {
            return {x * f, y * f, z * f};
        }

        Point operator+(const Point& other) const
        {
            return {x + other.x, y + other.y, z + other.z};
        }

        friend float dot(const Point& a, const Point& b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        template<typename T> friend T& operator<<(T& lhs, const Point& rhs)
        {
            lhs << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
            return lhs;
        }
    };

    struct Triangle
    {
        using ID = int;
        Point points[3];

        Triangle() = default;
        Triangle(const Point& a, const Point& b, const Point& c) : points{a, b, c} {}
    };

    using Mesh = std::vector<Triangle>;
    using MeshAsID = std::vector<Triangle::ID>;

    struct AABB
    {
        Point min, max;

        [[nodiscard]] bool inside(const Point& p) const
        {
            return p.x >= min.x && p.y >= min.y && p.z >= min.z
                && p.x < max.x  && p.y < max.y  && p.z < max.z;
        }
    };

    enum Side
    {
        NEAR, FAR
    };

    /**
     * Axis aligned straight line.
     */
    struct Line
    {
        /**
         * The position of the straight line in the init dimension.
         * As this is a straight line, it goes ad infinitum to the others 2 dimensions.
         */
        float p;

        /**
         * The dimension of the init.
         * 0 for X, 1 for Y, 2 for Z.
         */
        int dim;

        /**
         * Check on which side a point is from a straight line.
         * @param point The point to check.
         *
         * @return
         *      NEAR if the point is below the axe in the init dimension, otherwise FAR.
         *      For example, the straight line X=5.5 is represented by {p=5.5, dim=0}.
         *      Then for a point P=(3, 123, 456) query returns NEAR.
         *      For a point P'=(6, 987, 654) query returns FAR.
         */
        [[nodiscard]] Side query(const Point& point) const
        {
            return point[dim] < p ? NEAR : FAR;
        }
    };

public:
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
    static Point findClosestPointOnTriangle(const Point& query, const Triangle& triangle);

    /**
     * Check if a 3D point is inside a triangle
     */

    static void store_min(float& current, float newValue)
    {
        if(current > newValue) {
            current = newValue;
        }
    }

    static void store_max(float& current, float newValue)
    {
        if(current < newValue) {
            current = newValue;
        }
    }

    /**
     * @brief Split a set of triangles in near and far sets.
     *
     * @param mesh The mesh to split.
     * @param meshSize The count of triangles in the mesh.
     * @param axe The position of the split straight line.
     * @param outputsOrig
     *      Where to store the split mesh.
     *      outputs[NEAR] will store the near mesh.
     *      output[FAR] will store the far mesh.
     *      Some triangles can be stored on both sides.
     *      The array must be preallocated and have enough place.
     *      That means, each output should have at least the size of the input mesh because
     *      we don't know and maybe all triangles will belong to both children.
     *
     * @param outputSizes
     *      outputSizes[NEAR] will store the size of the near output mesh.
     *      outputSizes[NEAR] will store the size of the far output mesh.
     */
    void split(const Triangle::ID *mesh, int meshSize, const Line& axe, Triangle::ID* const outputsOrig[2], int outputSizes[2]);

public:
    KDTree() = default;
    explicit KDTree(Mesh mesh);

    struct NPQueryRet
    {
        Point point;
        Triangle::ID id;
    };

    /**
     * Compute the nearest point on the mesh from the query position.
     *
     * @param pos The query position.
     *
     * @return A pair (point, id) where id is the ID of the triangle.
     */
    [[nodiscard]] NPQueryRet findNearestPointOnMesh(const Point& pos) const;

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
        return (1 << (m_maxLevel + 1)) - 1;
    }

    template<typename T>
    T& summarize(T& out)
    {
        out << "KDTree summary:" << std::endl;
        out << "    Max level: " << m_maxLevel << std::endl;
        out << "    Max nodes: " << getMaxNodesCount() << std::endl;
        out << "    Non-null nodes: " << (m_totalLeafNodes * 2 - 1) << std::endl; // Handshaking Lemma
        out << "    Leaves: " << m_totalLeafNodes << std::endl;
        out << "    Triangles in leaves: " << m_leavesBuffer.size()
            << " (" << m_leavesBuffer.size() * sizeof(Triangle::ID) << " bytes)" << std::endl;

        return out;
    }

private:
    /**
     * @brief Represent each node of the k-d tree.
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
        Line line() const
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
        bool leaf() const
        {
            // left or right it doesn't matter
            return !header.hasChildren;
        }
    };

    using NodeID = int;

private:
    void init();

    void searchRecursive(const Point& pos, NodeID nodeID, float& currentDist, Triangle::ID& currentID, Point& currentPoint) const;

    Node& getRoot() { return m_nodes[0]; }

private:
    AABB m_rootAABB;
    std::vector<Triangle> m_mesh;
    std::vector<Node> m_nodes;
    std::vector<Triangle::ID> m_leavesBuffer; ///< Each part of the array stores one leave
    int m_maxLevel;
    int m_totalLeafNodes{0}; ///< Total count of leaves

    // CUDA buffers
    // Tehy are built at the end of the KDTree initialization
    thrust::device_vector<Triangle> m_d_mesh;
    thrust::device_vector<Node> m_d_nodes;
    thrust::device_vector<Triangle::ID> m_d_leavesBuffer;

    /**
     * Contains pointers to device memory.
     * Can be copied to CUDA kernel to access the KDTree.
     */
    class OnDevice
    {
        /**
         * Like `std::span` class on device.
         * There are no `thrust::span` class for now.
         */
        template<typename T>
        struct Span
        {
            T* data;
            size_t size;
        };

    private:
        AABB m_rootAABB;
        Span<Triangle> m_mesh;
        Span<Node> m_nodes;
        Span<Triangle::ID> m_leavesBuffer;
        int m_maxLevel;
        int m_totalLeafNodes;
    };
};