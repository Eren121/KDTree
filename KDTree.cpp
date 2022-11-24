#include "KDTree.h"
#include <climits>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <cmath>

float KDTree::median(std::vector<float> vec)
{
    size_t size = vec.size();

    if (size == 0)
    {
        return 0;  // Undefined, really.
    }
    else
    {
        std::sort(vec.begin(), vec.end());
        if (size % 2 == 0)
        {
            return (vec[size / 2 - 1] + vec[size / 2]) / 2;
        }
        else
        {
            return vec[size / 2];
        }
    }
}

KDTree::AABB KDTree::computeBoundingBox(const std::vector<Point>& points)
{
    AABB res;

    if (!points.empty())
    {
        const auto& firstPoint = points.front();

        // Initialize the bounding box to a point
        for (int dim = 0; dim < 3; dim++)
        {
            res.min[dim] = firstPoint[dim];
            res.max[dim] = firstPoint[dim];
        }

        // Grow the bounding box for each point if needed
        for (const Point& point: points)
        {
            for (int dim = 0; dim < 3; dim++)
            {
                store_min(res.min[dim], point[dim]);
                store_max(res.max[dim], point[dim]);
            }
        }
    }

    return res;
}

KDTree::KDTree(const std::vector<Point>& points)
{
    m_rootAABB = computeBoundingBox(points);

    m_root = std::make_unique<Node>();

    std::stack<SplitStack> stack;


    stack.push(SplitStack{0, points, m_root.get(), m_rootAABB});

    split(stack);
}

void KDTree::split(std::stack<SplitStack>& stack)
{
    // Split recursively in x, y, z, x, y, z...
    // Split at the center

    // dim axis -->

    // 0 --------- aabb[dim].min --------------------------- aabb[dim].max --------- +inf
    // ------------------|--------------------|-------------------|-------------------
    // ---------------------- left node ----------- right node -----------------------

    //                    <----------------->    <--------------->
    // splitDistance:        if left                 if right

    while (!stack.empty())
    {
        std::vector<Point> points = std::move(stack.top().points);
        Node& node = *stack.top().node;
        int dim = stack.top().dim;
        AABB aabb = stack.top().aabb;
        stack.pop();


        // Stop condition
        if (points.size() > 100)
        {
            node.splitDim = dim;

            // Absolute position in the dimension of the split
            node.splitDistance = (aabb.max[dim] + aabb.min[dim]) / 2.0f;

            AABB leftAABB = aabb;
            leftAABB.max[dim] = node.splitDistance;

            AABB rightAABB = aabb;
            rightAABB.min[dim] = leftAABB.max[dim];

            std::vector<Point> leftPoints, rightPoints;

            for (const Point& p: points)
            {
                if (leftAABB.inside(p))
                {
                    leftPoints.push_back(p);
                }
                else
                {
                    rightPoints.push_back(p);
                }
            }

            const int nextDim = (dim + 1) % 3;

            node.right = std::make_unique<Node>();
            stack.push(SplitStack{nextDim, std::move(rightPoints), node.right.get(), rightAABB});

            node.left = std::make_unique<Node>();
            stack.push(SplitStack{nextDim, std::move(leftPoints), node.left.get(), leftAABB});
        }
        else
        {
            // Leaf
            node.points = std::move(points);
        }
    }
}

KDTree::Point KDTree::computeNearestNeighbor(const KDTree::Point& pos) const
{
    // Are we left or right?

    const Node *node = m_root.get();
    AABB aabb = m_rootAABB;

    float dist = FLT_MAX;
    Point res;
    searchRecursive(pos, m_root.get(), dist, res);

    return res;
}

void KDTree::searchRecursive(const Point& pos, Node *node, float& currentDist, Point& currentNeighbor) const
{
    // Are we on a leaf?
    if (node->leaf())
    {
        // We are on a leaf
        // Search brute force into the leaf node
        for (const auto& other: node->points)
        {
            const float d = other.distanceSquared(pos);
            if (d < currentDist)
            {
                currentDist = d;
                currentNeighbor = other;
            }
        }
    }
    else
    {
        Node *front, *back;

        // Are we on the left side?
        if (pos[node->splitDim] < node->splitDistance)
        {
            // Pos is on the left side
            front = node->left.get();
            back = node->right.get();
        }
        else
        {
            // Pos is on the right side
            front = node->right.get();
            back = node->left.get();
        }

        searchRecursive(pos, front, currentDist, currentNeighbor);

        // If the current closest point is closer than the closest point of the back face, no need to search in the back
        // face because it will be always further.
        // If not, we save half of the time for the current node
        const float backDist = fabsf(node->splitDistance - pos[node->splitDim]);
        // Do not forget all distances all squared
        if (backDist * backDist <= currentDist)
        {
            // If it can be closer, search also in this node
            searchRecursive(pos, back, currentDist, currentNeighbor);
        }
    }
}
