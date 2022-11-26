#include "KDTree.h"
#include <climits>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>
#include <numeric>

namespace
{
    template<typename... T>
    void trace(T&&... args)
    {
#if KDTREE_TRACE
        ((std::cout << args << " "), ...) << std::endl;
#endif
    }
}

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
    : m_rootAABB(computeBoundingBox(m_mesh)),
      m_mesh(std::move(mesh)),
      m_maxLevel(15)
{
    init();
}

void KDTree::init()
{
    const size_t bytesBuffer = 10'000'000; // 10MB

    // Allocate the maximum possible size for the leaves node
    // If it is outreached then undefined behaviour; pointers may be invalided on reallocation
    // And leaves mesh points to garbage
    m_leavesBuffer.reserve(bytesBuffer / sizeof(m_leavesBuffer[0]));

    // We can't use recursive functions, because the OS stack is too small for our needs in recursion.
    // Treat the members as the parameters of the recursive function.
    // Also, we iterate BFS and not DFS
    struct Task
    {
        int dim; ///< The dimension to the split
        NodeID nodeID; ///< The node, allocated, to fill
        AABB aabb; ///< The bounding box of the node.
        int level; ///< Current level of depth. Used for termination condition.
        Triangle::ID *inputMesh; ///< Remaining candidates for this node
        int inputMeshSize; ///< Count of input mesh for this node
        Triangle::ID *outputMesh; ///< Preallocated output for this mesh split, of size x2 the count of triangles
    };


    // Allocate the maximum number of node
    // Also initialize to zero (empty node)
    m_nodes.resize(getMaxNodesCount());

    // We use a vector just because we need the clear() method which is more effecient
    // than a pop() loop, or '= {}' because it may reallocate memory
    std::vector<Task> stack;

    // Split recursively in x, y, z, x, y, z...
    // Split at the center

    // dim axis -->

    // 0 --------- aabb[dim].min --------------------------- aabb[dim].max --------- +inf
    // ------------------|--------------------|-------------------|-------------------
    // ---------------------- left node ----------- right node -----------------------

    //                    <----------------->    <--------------->
    // splitDistance:        if left                 if right

    #pragma omp parallel default(none) shared(stack)
    #pragma omp single
    {
        // Generator (master) thread current level
        int generatorLevel = 0;

        // The temporary working buffer to store the triangles for the current level (input),
        // And the next level (output)
        // We don't know a reasonable upper bound of the size of the vectors, or a very high upper bounds impracticable
        // to reserve: the level L has at maximum 2^L * mesh.size() triangles, which is very high for deep levels.
        std::vector<Triangle::ID> inputTriangles;
        std::vector<Triangle::ID> outputTriangles;

        // To split the task children, allocate memory at each level only once globally for the triangles
        // Each task has a unique array associated to it starting at task.inputMesh pointing in same
        // memory inside inputTriangles.
        // There is as well an output buffer mesh.outputMesh pointing inside an offset in outputTriangles.
        // This temporary working "buffer" for the task is of size 2x the count of triangles of the task
        // That means the working buffer will always have enough space to store the children triangles even
        // if both have all the mesh (as a note, this case is very inefficient). We could optimize it and stop
        // Recursion for example if more than X% of triangles are on both side.
        // The working buffer is split in two contiguous array to store the children near and far triangles.
        // There is no race condition on the input variable because each task may only access its own part of the array.
        // There is also no race condition on the output because each task has its own part to the output array.

        // At first, there is only one task (the root)
        // And all the triangles are contained once in the root node
        inputTriangles.resize(m_mesh.size());
        std::iota(inputTriangles.begin(), inputTriangles.end(), 0); // Initial candidates is the entire mesh

        // At the very most both near and far children of the root store all the mesh
        outputTriangles.resize(inputTriangles.size() * 2);

        {
            // First task
            stack.push_back({
                .dim = 0,
                .nodeID = 0,
                .aabb = m_rootAABB,
                .level = 0,
                .inputMesh = &inputTriangles[0],
                .inputMeshSize = static_cast<int>(inputTriangles.size()),
                .outputMesh = &outputTriangles[0]
            });
        }

        while(!stack.empty())
        {
            // Queue the stack tasks at once, then wait all
            trace("Spawn", stack.size(), "tasks for level", generatorLevel);

            // Run all tasks
            // First get all tasks to avoid a race condition on the stack,
            // because a task may push() to the stack, possibly immediately
            auto tasks = std::move(stack);
            stack.clear();

            // Total count of triangles outputs for the current level, which is also
            // the total count of input triangles for the next level.
            // Filled as things progress by the tasks (atomically to avoid race condition)
            // Also permit to know where the offset should be for each task in the output.
            int totalOutputTriangles = 0;

            #pragma omp taskloop default(none) shared(tasks, stack, totalOutputTriangles)
            for(int t = 0; t < tasks.size(); t++)
            {
                auto& task = tasks[t];
                const NodeID nodeID = task.nodeID;
                Node& node = m_nodes[task.nodeID];
                const int dim = task.dim;
                const AABB aabb = task.aabb;
                const int level = task.level;

                // Stop condition
                // FOR TRIANGLES: it's not guaranteed we can have less a given number of triangles, so we always should
                // Stop on a max. level
                if(task.inputMeshSize > 100 && level < m_maxLevel)
                {
                    // Node must be split, split here

                    node.header.dim = dim;
                    node.header.hasChildren = true;

                    // We split at the center of the parent AABB
                    node.p = (aabb.max[dim] + aabb.min[dim]) / 2.0f;

                    AABB aabbs[2];
                    aabbs[NEAR] = aabb;
                    aabbs[NEAR].max[dim] = node.p;
                    aabbs[FAR] = aabb;
                    aabbs[FAR].min[dim] = aabbs[NEAR].max[dim];

                    Triangle::ID *outputs[2];
                    outputs[NEAR] = &task.outputMesh[0];
                    outputs[FAR] = &task.outputMesh[task.inputMeshSize];

                    int outputSizes[2];

                    // COSTLY SPLIT in preallocated memory
                    split(task.inputMesh, task.inputMeshSize, node.line(), outputs, outputSizes);

                    const int nextDim = (dim + 1) % 3;

                    NodeID childrenIDs[2];
                    childrenIDs[NEAR] = 2 * nodeID + 1; // "Left child" (near)
                    childrenIDs[FAR] = 2 * nodeID + 2; // "Right child" (far)

                    #pragma omp critical
                    {
                        for(int s = 0; s < 2; s++) // for NEAR and FAR
                        {
                            // We don't know yet the child outputMesh BECAUSE
                            // The memory is not yet allocated,
                            // We have to know how many triangles in total there is for the current level for that, which is only
                            // know when all the tasks will be completed.
                            // When all tasks for the current level will be completed, the master thread
                            // Will allocate a new buffer with this size (totalOutputTriangles final value x2).

                            totalOutputTriangles += outputSizes[s];

                            stack.push_back({
                                .dim = nextDim,
                                .nodeID = childrenIDs[s],
                                .aabb = aabbs[s],
                                .level = level + 1,
                                .inputMesh = outputs[s],
                                .inputMeshSize = outputSizes[s], // Filling that the master thread will also know which offset to give to outputMesh

                                // WILL BE UPDATED BY MASTER THREAD
                                .outputMesh = nullptr
                            });
                        }
                    }
                }
                else
                {
                    // Leaf node
                    // Store the final mesh in the leaf node

                    #pragma omp critical
                    {
                        const auto offset = m_leavesBuffer.size();

                        // Reserve memory in the leaves buffer
                        m_leavesBuffer.insert(m_leavesBuffer.end(), task.inputMesh, task.inputMesh + task.inputMeshSize);
                        m_totalLeafNodes++;

                        // We consider the leavesBuffer has enouhgh size and inserting won't invalidate pointers
                        node.mesh = &m_leavesBuffer[offset];
                        node.meshSize = task.inputMeshSize;

                        // DO NOT increment totalOutputTriangles
                        // Because this variable is used to compute the next output size,
                        // but as this is a leaf there is no child node so no need for output for this node for the next level.
                    }
                }
            }

            #pragma omp taskwait

            trace("Output buffer size: ", outputTriangles.size() * sizeof(outputTriangles[0]));

            // Double buffering of temporary split buffer
            using std::swap;
            swap(inputTriangles, outputTriangles);

            // Allocate the next output buffer
            // We don't care of the content as it will be overwritten,
            // if there is enough space no reallocation will occur wich is good
            // The size of the next output buffer is upper bounded by twice the count of next total inputs.
            outputTriangles.resize(totalOutputTriangles * 2);

            {
                // For each task, associate a unique offset in the new output buffer

                int offset = 0; // Offset relative tot totalOutputTriangles

                for(int i = 0; i < stack.size(); i++)
                {
                    auto& task = stack[i];
                    task.outputMesh = &outputTriangles[offset * 2];

                    offset += task.inputMeshSize;
                }

                // May be useful in parallelism, to spot some problem with race condition, like if one increment drop...
                assert(offset == totalOutputTriangles);
            }

            generatorLevel++;
        }
    }
}


KDTree::NPQueryRet KDTree::findNearestPointOnMesh(const Point& pos) const
{
    // Are we near or far?
    NPQueryRet ret{};

    float currentDist = FLT_MAX;

    searchRecursive(pos, 0, currentDist, ret.id, ret.point);

    return ret;
}

void KDTree::searchRecursive(const Point& pos, NodeID nodeID, float& currentDist, Triangle::ID& currentID,
                             Point& currentPoint) const
{
    const Node& node = m_nodes[nodeID];

    // Are we on a leaf?
    if(node.leaf())
    {
        // We are on a leaf
        // Search brute force into the leaf node
        for(int i = 0; i < node.meshSize; i++)
        {
            const auto& triangleID = node.mesh[i];
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
        NodeID front, back;
        const Line& split = node.line();

        NodeID childrenIDs[2];
        childrenIDs[NEAR] = 2 * nodeID + 1; // "Left child" (near)
        childrenIDs[FAR] = 2 * nodeID + 2; // "Right child" (far)

        // Which side I am?
        switch(split.query(pos))
        {
            case NEAR:
                // Pos is on the near side
                front = childrenIDs[NEAR];
                back = childrenIDs[FAR];
                break;

            case FAR:
                // Pos is on the far side
                front = childrenIDs[FAR];
                back = childrenIDs[NEAR];
                break;
        }

        searchRecursive(pos, front, currentDist, currentID, currentPoint);

        // If the current closest point is closer than the closest point of the back face, no need to search in the back
        // face because it will be always further.
        // If so, we save half of the time for the current node
        const float backDist = fabsf(split.p - pos[split.dim]);
        // Do not forget currentDist is squared
        if(backDist * backDist <= currentDist)
        {
            // If it can be closer, search also in this node
            searchRecursive(pos, back, currentDist, currentID, currentPoint);
        }
    }
}

void KDTree::split(const Triangle::ID *mesh, int meshSize, const Line& axe, Triangle::ID* const outputsOrig[2], int outputSizes[2])
{
    // Save locally to not modify original pointer
    Triangle::ID* outputs[2];
    outputs[NEAR] = outputsOrig[NEAR];
    outputs[FAR] = outputsOrig[FAR];


    // Iterate all triangles,
    // We can't split them as they can belong to both sides
    for(int i = 0; i < meshSize; i++)
    {
        const auto& triangleID = mesh[i];
        const auto& triangle = m_mesh[triangleID];

        // If all points of the triangle are on one side, the triangle is not colliding with the other side (because
        // triangle and AABB are convex shapes).

        const auto side = axe.query(triangle.points[0]);
        if(side == axe.query(triangle.points[1]) && side == axe.query(triangle.points[2]))
        {
            // All points are on the same side
            // So the triangle is on one side
            *(outputs[side]++) = (triangleID);
        }
        else
        {
            // All points are not on the same side
            // So the triangle is one both side
            *(outputs[NEAR]++) = (triangleID);
            *(outputs[FAR]++) = (triangleID);
        }
    }

    for(int i = 0; i < 2; i++)
    {
        outputSizes[i] = static_cast<int>((outputs[i] - outputsOrig[i]));
    }
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