#include "KDTree.h"
#include <climits>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>

KDTree::AABB KDTree::computeBoundingBox(const Mesh& mesh)
{
    // Computing the AABB of a mesh is just the AABB of all points because an AABB is convex

    AABB res{};

    if(!mesh.empty())
    {
        const auto& firstPoint = mesh[0].points[0];

        // Initialize the bounding box to a point
        for(int dim = 0; dim < 3; dim++)
        {
            res.min[dim] = firstPoint[dim];
            res.max[dim] = firstPoint[dim];
        }

        // Grow the bounding box for each point if needed
        for(const auto& triangle: mesh)
        {
            for(const auto& point: triangle.points)
            {
                for(int dim = 0; dim < 3; dim++)
                {
                    store_min(res.min[dim], point[dim]);
                    store_max(res.max[dim], point[dim]);
                }
            }
        }
    }

    return res;
}

KDTree::KDTree(Mesh mesh)
    : m_mesh(std::move(mesh)),
      m_rootAABB(computeBoundingBox(m_mesh)),
      m_root(std::make_unique<Node>())
{
    init();
}

void KDTree::init()
{
    std::stack<SplitStack> stack;

    {
        // Initial candidates is the entire mesh
        MeshAsID candidates;
        candidates.reserve(m_mesh.size());

        for(int i = 0; i < m_mesh.size(); i++)
        {
            candidates.push_back(i);
        }

        stack.push(SplitStack{0, std::move(candidates), m_root.get(), m_rootAABB, 0});
    }

    // Split recursively in x, y, z, x, y, z...
    // Split at the center

    // dim axis -->

    // 0 --------- aabb[dim].min --------------------------- aabb[dim].max --------- +inf
    // ------------------|--------------------|-------------------|-------------------
    // ---------------------- left node ----------- right node -----------------------

    //                    <----------------->    <--------------->
    // splitDistance:        if left                 if right

    #pragma omp parallel default(none) shared(stack, std::cout)
    {
        #pragma omp single
        while(!stack.empty())
        {
            // Queue the stack tasks at once, then wait all
            std::cout << "Spawn " << stack.size() << " tasks" << std::endl;

            while(!stack.empty())
            {
                SplitStack arg;

                #pragma omp critical
                {
                    arg = std::move(stack.top());
                    stack.pop();
                }

                #pragma omp task default(none) firstprivate(arg) shared(stack)
                {
                    auto mesh = std::move(arg.mesh);
                    Node& node = *arg.node;
                    const int dim = arg.dim;
                    const AABB aabb = arg.aabb;
                    const int level = arg.level;

                    constexpr int MAX_LEVEL = 5;

                    // Stop condition
                    // FOR TRIANGLES: it's not guaranteed we can have less a given number of triangles, so we always should
                    // Stop on a recursive condition
                    // We can also stop it if the next split don't split enough, let's say more than 50% of triangles are on both sides
                    if(mesh.size() > 100 && level < MAX_LEVEL)
                    {
                        node.split.dim = dim;

                        // We split at the center of the parent AABB
                        node.split.p = (aabb.max[dim] + aabb.min[dim]) / 2.0f;

                        AABB nearAABB = aabb;
                        nearAABB.max[dim] = node.split.p;

                        AABB farAABB = aabb;
                        farAABB.min[dim] = nearAABB.max[dim];

                        auto [near, far] = split(mesh, node.split);

                        const int nextDim = (dim + 1) % 3;

                        #pragma omp critical
                        {
                            node.far = std::make_unique<Node>();
                            stack.push(SplitStack{nextDim, std::move(far), node.far.get(), farAABB, level + 1});

                            node.near = std::make_unique<Node>();
                            stack.push(SplitStack{nextDim, std::move(near), node.near.get(), nearAABB, level + 1});
                        }
                    }
                    else
                    {
                        // Leaf
                        node.mesh = std::move(mesh);
                    }
                }
            }

            #pragma omp taskwait
        }
    }
}


KDTree::NPQueryRet KDTree::findNearestPointOnMesh(const Point& pos) const
{
    // Are we near or far?
    NPQueryRet ret{};

    float currentDist = FLT_MAX;

    searchRecursive(pos, m_root.get(), currentDist, ret.id, ret.point);

    return ret;
}

void KDTree::searchRecursive(const Point& pos, Node *node, float& currentDist, Triangle::ID& currentID,
                             Point& currentPoint) const
{
    // Are we on a leaf?
    if(node->leaf())
    {
        // We are on a leaf
        // Search brute force into the leaf node
        for(const auto& triangleID: node->mesh)
        {
            const auto nearestPtOnTriangle = findClosestPointOnTriangle(pos, m_mesh[triangleID]);
            const float d = nearestPtOnTriangle.distanceSquared(pos);
            if(d < currentDist)
            {
                currentDist = d;
                currentID = triangleID;
                currentPoint = nearestPtOnTriangle;
            }
        }
    }
    else
    {
        Node *front, *back;

        // Which side I am?
        switch(node->split.query(pos))
        {
            case NEAR:
                // Pos is on the near side
                front = node->near.get();
                back = node->far.get();
                break;

            case FAR:
                // Pos is on the far side
                front = node->far.get();
                back = node->near.get();
                break;
        }

        searchRecursive(pos, front, currentDist, currentID, currentPoint);

        // If the current closest point is closer than the closest point of the back face, no need to search in the back
        // face because it will be always further.
        // If so, we save half of the time for the current node
        const float backDist = fabsf(node->split.p - pos[node->split.dim]);
        // Do not forget currentDist is squared
        if(backDist * backDist <= currentDist)
        {
            // If it can be closer, search also in this node
            searchRecursive(pos, back, currentDist, currentID, currentPoint);
        }
    }
}

std::pair<KDTree::MeshAsID, KDTree::MeshAsID> KDTree::split(const MeshAsID& mesh, const Line& axe)
{
    MeshAsID parts[2];
    
    // Iterate all triangles,
    // We can't split them as they can belong to both sides
    for(const auto& triangleID : mesh)
    {
        const auto& triangle = m_mesh[triangleID];

        // If all points of the triangle are on one side, the triangle is not colliding with the other side (because
        // triangle and AABB are convex shapes).

        const auto side = axe.query(triangle.points[0]);
        if(side == axe.query(triangle.points[1]) && side == axe.query(triangle.points[2]))
        {
            // All points are on the same side
            // So the triangle is on one side
            parts[side].push_back(triangleID);
        }
        else
        {
            // All points are not on the same side
            // So the triangle is one both side
            parts[NEAR].push_back(triangleID);
            parts[FAR].push_back(triangleID);
        }
    }

    return {parts[NEAR], parts[FAR]};
}

KDTree::Point KDTree::findClosestPointOnTriangle(const KDTree::Point& query, const KDTree::Triangle& triangle)
{
    // https://stackoverflow.com/a/32255438/5110937

    using std::clamp;

    auto edge0 = triangle.points[1] - triangle.points[0];
    auto edge1 = triangle.points[2] - triangle.points[0];
    auto v0 = triangle.points[0] - query;

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0);
    float e = dot(edge1, v0);

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if(s + t < det)
    {
        if(s < 0.f)
        {
            if(t < 0.f)
            {
                if(d < 0.f)
                {
                    s = clamp(-d / a, 0.f, 1.f);
                    t = 0.f;
                }
                else
                {
                    s = 0.f;
                    t = clamp(-e / c, 0.f, 1.f);
                }
            }
            else
            {
                s = 0.f;
                t = clamp(-e / c, 0.f, 1.f);
            }
        }
        else if(t < 0.f)
        {
            s = clamp(-d / a, 0.f, 1.f);
            t = 0.f;
        }
        else
        {
            float invDet = 1.f / det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if(s < 0.f)
        {
            float tmp0 = b + d;
            float tmp1 = c + e;
            if(tmp1 > tmp0)
            {
                float numer = tmp1 - tmp0;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1 - s;
            }
            else
            {
                t = clamp(-e / c, 0.f, 1.f);
                s = 0.f;
            }
        }
        else if(t < 0.f)
        {
            if(a + d > b + e)
            {
                float numer = c + e - b - d;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1 - s;
            }
            else
            {
                s = clamp(-e / c, 0.f, 1.f);
                t = 0.f;
            }
        }
        else
        {
            float numer = c + e - b - d;
            float denom = a - 2 * b + c;
            s = clamp(numer / denom, 0.f, 1.f);
            t = 1.f - s;
        }
    }

    return triangle.points[0] + edge0 * s + edge1 * t;
}