#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <stack>

/**
 * @brief K-d tree of 3D float points.
 *
 * @see Variable names, and algorithm inspired by https://youtu.be/TrqK-atFfWY?t=2567
 */
class KDTree
{
public:
    /**** Utility structures ****/
    struct Point
    {
        float x, y, z;

        float& operator[](int i)
        {
            assert(i < 3 && i >= 0);

            switch(i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
            }

            throw std::runtime_error("Invalid index");
        }

        const float& operator[](int i) const
        {
            return const_cast<Point&>(*this)[i];
        }

        float distance(const Point& other) const
        {
            return sqrtf(distanceSquared(other));
        }

        float distanceSquared(const Point& other) const
        {
            const float dx = (x - other.x);
            const float dy = (y - other.y);
            const float dz = (z - other.z);
            return dx*dx + dy*dy + dz*dz;
        }

        template<typename T> friend T& operator<<(T& lhs, const Point& rhs)
        {
            lhs << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
            return lhs;
        }
    };

    struct AABB
    {
        Point min, max;

        bool inside(const Point& p) const
        {
            return p.x >= min.x && p.y >= min.y && p.z >= min.z
                && p.x < max.x  && p.y < max.y  && p.z < max.z;
        }
    };

public:
    /**** Utility (static) functions ****/

    /**
     * @brief Compute the bounding box of a set of points.
     * @param points
     *      The points to make the boudning box from.
     *      If the set of points is empty, the returned value is undefined.
     * @return
     *      The bounding box of theses points.
     *      The bounding box will be always of minimum size, i.e. there will be a vertex on each of the 6 faces of the
     *      returned AABB.
     */
    static AABB computeBoundingBox(const std::vector<Point> &points);

    static float median(std::vector<float> vec);

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


public:
    KDTree() = default;
    explicit KDTree(const std::vector<Point>& points);

    Point computeNearestNeighbor(const Point& pos) const;

private:
    /**
     * @brief Represent each node of the k-d tree.
     */
    struct Node
    {
        /**
         * @brief
         *      Children. Both of them or neither of them are null.
         */
        std::unique_ptr<Node> left, right;

        /**
         * The distance between the origin and the wall to split (for left node),
         * or the distance from the wall and the end to split (for right node).
         */
        float splitDistance;
        int splitDim;

        std::vector<Point> points;

        /**
         * @return true if this node is a leaf node (it has no children).
         */
        bool leaf() const
        {
            // left or right it doesn't matter
            return left == nullptr;
        }
    };

    struct SplitStack
    {
        int dim;
        std::vector<Point> points;
        Node* node;
        AABB aabb;
    };

private:
    /**
     * @brief Split the tree during the build.
     * @param dim The dimension to split.
     * @param points The list of remaining candidate points for this area, inside the AABB.
     * @param node The node, allocated, to fill.
     * @param aabb The bounding box of the node.
     */
    void split(std::stack<SplitStack>& stack);

    void searchRecursive(const Point& pos, Node* node, float& currentDist, Point& currentNeighbor) const;

private:
    AABB m_rootAABB;
    std::unique_ptr<Node> m_root;
};
