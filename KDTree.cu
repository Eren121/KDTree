#include "KDTree.cuh"
#include <climits>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>
#include <numeric>
#include <thrust/host_vector.h>

namespace
{
#if KDTREE_TRACE

    template<typename... T>
    void trace(T&& ... args)
    {
        ((std::cout << args << " "), ...) << std::endl;
    }

    template<typename T>
    void print_vector(const std::string& title, const thrust::device_vector<T>& t)
    {
        thrust::host_vector<T> v(t);

        std::cout << title << ": ";
        for(int i = 0; i < v.size(); i++)
        {
            std::cout << v[i] << ", ";
        }
        std::cout << std::endl;
    }

#else
    template<typename... T> void trace(T&&...) {}
    template<typename... T> void print_vector(T&&...) {}
#endif
}

AABB KDTreeBase::computeBoundingBox(const Mesh& mesh)
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

KDTree::KDTree(const Mesh& mesh, Heuristics heuristics)
    : m_mesh(mesh),
      m_rootAABB(computeBoundingBox(mesh)),
      m_heuristics(heuristics)
{
    init();
}

namespace
{
    /**
     * For the original version on the CPU,
     * We can't use recursive functions, because the OS stack is too small for our needs in recursion.
     * Its also more optimized to not use recursive functions (no context argument needed).
     *
     * Also for the GPU version, we offload each task on the GPU.
     * Treat the members as the parameters of the recursive function.
     * Also, we iterate BFS and not DFS.
     */
    struct TaskData
    {
        int dim; ///< The dimension of the split
        KDTree::NodeID nodeID; ///< The node, to fill
        AABB aabb; ///< The bounding box of the node.
        int level; ///< Current level of depth. Used for termination condition.
        int inputMeshOff; ///< Remaining candidates for this node, pointing to TaskOp::inputMesh
        int inputMeshSize; ///< Count of input mesh for this node
        int outputMeshOff; ///< Preallocated output for this mesh split, pointing to TaskOp::outputMesh,, of size x2 inputMeshSize, first half is near node, second half is far node
    };

    /**
     * Thrust operation to fill the current level.
     * All member pointers point to the device memory.
     */
    struct TaskOp : KDTreeBase
    {
        /**
         * Pointer to all the output tasks.
         * Output array filled to store the new/outputs/children tasks.
         * It has always enough place (x2 size of input tasks).
         * It should be synchronized to avoid data race.
         * You can just fetch-and-add (atomicAdd in CUDA) the current size to reserve some offset in the array.
         * The size is just a single integer, as a pointer to save it on the host side.
         */
        TaskData *outputTasks;
        int *outputTasksSize;

        /**
         * Front of the KDTree::m_leavesBuffer array.
         * Like outputTasks, filling this vector should be synchronized with fetch-and-add.
         * The size is just a single integer, as a pointer to save it on the host side.
         */
        Triangle::ID *leavesBuffer;
        int *leavesBufferSize;

        /**
         * Front of the KDTree::m_node array.
         * No need for synchronization, each kernel operates on a different node.
         */
        Node *nodes;

        /**
         * Same as the local variable in KDTree::init().
         * Temporary buffers during the tree construction.
         */
        const Triangle::ID *inputTriangles;
        Triangle::ID *outputTriangles;

        /**
         * Same as the local variable in KDTree::init().
         * Should be synchronized.
         */
        int *totalOutputTriangles;

        /**
         * Same as KDTree::m_mesh
         */
        const Triangle *mesh;

        Heuristics heuristics;

        /**
         * @brief Split a set of triangles in near and far sets.
         *
         * @param subMesh The mesh to split.
         * @param subMeshSize The count of triangles in the mesh.
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
        CUDA_BOTH void
        split(const Triangle::ID *subMesh, int subMeshSize, const Line& axe, Triangle::ID *const outputsOrig[2],
              int outputSizes[2]);

        __device__ void operator()(const TaskData& taskData);
    };
}

void KDTree::init()
{
    const size_t bytesBuffer = 10 * MB;

    // Allocate the maximum possible size for the leaves node
    // If it is outreached then undefined behaviour; pointers may be invalided on reallocation
    // And leaves mesh will point to undefined value
    m_leavesBuffer.resize(bytesBuffer / sizeof(m_leavesBuffer[0]));


    // Allocate the maximum number of node
    // Also initialize to zero (empty node)
    m_nodes.resize(getMaxNodesCount());

    // We use a vector just because we need the clear() method which is more effecient
    // than a pop() loop, or '= {}' because it may reallocate memory
    thrust::device_vector<TaskData> inputTasks;

    // Split recursively in x, y, z, x, y, z...
    // Split at the center

    // dim axis -->

    // 0 --------- aabb[dim].min --------------------------- aabb[dim].max --------- +inf
    // ------------------|--------------------|-------------------|-------------------
    // ---------------------- left node ----------- right node -----------------------

    //                    <----------------->    <--------------->
    // splitDistance:        if left                 if right

    {
        // Generator (master) thread current level
        int generatorLevel = 0;

        // The temporary working buffer to store the triangles for the current level (input),
        // And the next level (output)
        // We don't know a reasonable upper bound of the size of the vectors, or a very high upper bounds impracticable
        // to reserve: the level L has at maximum 2^L * mesh.size() triangles, which is very high for deep levels.
        thrust::device_vector<Triangle::ID> inputTriangles;
        thrust::device_vector<Triangle::ID> outputTriangles;

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
        thrust::sequence(inputTriangles.begin(), inputTriangles.end(), 0); // Initial candidates is the entire mesh

        // At the very most both near and far children of the root store all the mesh
        outputTriangles.resize(inputTriangles.size() * 2);

        {
            // First task
            inputTasks.push_back(TaskData{
                .dim = 0,
                .nodeID = 0,
                .aabb = m_rootAABB,
                .level = 0,
                .inputMeshOff = 0,
                .inputMeshSize = static_cast<int>(inputTriangles.size()),
                .outputMeshOff = 0
            });
        }

        thrust::device_vector<TaskData> outputTasks;

        // Array of one element as a managed data on the device
        thrust::device_vector<int> leavesBufferSize(1, 0);

        while(!inputTasks.empty())
        {
            // Run all the stack tasks at once, then wait all
            trace("Spawn", inputTasks.size(), "tasks for level", generatorLevel);
            print_vector("inputTriangles", inputTriangles);

            // Run all tasks
            // First get all tasks to avoid a race condition on the stack,
            // because a task may push() to the stack, possibly immediately
            // At maximum each task will generate 2 new tasks (if non-leaf node)
            outputTasks.resize(inputTasks.size() * 2);
            thrust::device_vector<int> outputTasksSize(1, 0);

            // Total count of triangles outputs for the current level, which is also
            // the total count of input triangles for the next level.
            // Filled as things progress by the tasks (atomically to avoid race condition)
            // Also permit to know where the offset should be for each task in the output.
            // Just use thrust::device_vector of size 1 like a std::unique_ptr on the device to manage memory
            thrust::device_vector<int> totalOutputTriangles(1, 0);

            TaskOp op;
            op.outputTasks = thrust::raw_pointer_cast(outputTasks.data());
            op.outputTasksSize = thrust::raw_pointer_cast(outputTasksSize.data());
            op.leavesBuffer = thrust::raw_pointer_cast(m_leavesBuffer.data());
            op.leavesBufferSize = thrust::raw_pointer_cast(leavesBufferSize.data());
            op.nodes = thrust::raw_pointer_cast(m_nodes.data());
            op.inputTriangles = thrust::raw_pointer_cast(inputTriangles.data());
            op.outputTriangles = thrust::raw_pointer_cast(outputTriangles.data());
            op.totalOutputTriangles = thrust::raw_pointer_cast(totalOutputTriangles.data());
            op.mesh = thrust::raw_pointer_cast(m_mesh.data());
            op.heuristics = m_heuristics;

            thrust::for_each(inputTasks.begin(), inputTasks.end(), op);

            trace("Output buffer size: ", outputTriangles.size() * sizeof(outputTriangles[0]));

            // Double buffering of temporary split buffer
            using std::swap;
            swap(inputTriangles, outputTriangles);

            outputTasks.resize(outputTasksSize[0]);
            swap(inputTasks, outputTasks);

            // Allocate the next output buffer
            // We don't care of the content as it will be overwritten,
            // if there is enough space no reallocation will occur wich is good
            // The size of the next output buffer is upper bounded by twice the count of next total inputs.
            outputTriangles.resize(totalOutputTriangles[0] * 2);

            generatorLevel++;
        }

        m_leavesBuffer.resize(leavesBufferSize[0]);
    }
}

CUDA_BOTH
void TaskOp::split(const Triangle::ID *subMesh, int subMeshSize, const Line& axe, Triangle::ID *const outputsOrig[2],
                   int outputSizes[2])
{
    // Save locally to not modify original pointer
    Triangle::ID *outputs[2];
    outputs[NEAR] = outputsOrig[NEAR];
    outputs[FAR] = outputsOrig[FAR];


    // Iterate all triangles,
    // We can't split them as they can belong to both sides
    for(int i = 0; i < subMeshSize; i++)
    {
        const auto& triangleID = subMesh[i];
        const Triangle& triangle = mesh[triangleID];

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

__device__
void TaskOp::operator()(const TaskData& taskData)
{
    const NodeID nodeID = taskData.nodeID;
    Node& node = nodes[nodeID];
    const int dim = taskData.dim;
    const AABB aabb = taskData.aabb;
    const int level = taskData.level;

    // Stop condition
    // FOR TRIANGLES: it's not guaranteed we can have less a given number of triangles, so we always should
    // Stop on a max. level
    if(taskData.inputMeshSize > heuristics.maxNodeSize && level < heuristics.maxLevel)
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

        int outputsOff[2];
        outputsOff[NEAR] = taskData.outputMeshOff;
        outputsOff[FAR] = taskData.outputMeshOff + taskData.inputMeshSize;

        Triangle::ID *outputs[2];
        outputs[NEAR] = &outputTriangles[outputsOff[NEAR]];
        outputs[FAR] = &outputTriangles[outputsOff[FAR]];

        int outputSizes[2];

        // COSTLY SPLIT in preallocated memory
        split(&inputTriangles[taskData.inputMeshOff], taskData.inputMeshSize, node.line(), outputs,
              outputSizes);

        int nextDim;
        switch(heuristics.dim)
        {
            case Heuristics::DIM_2D:
                nextDim = (dim + 1) % 2;
                break;

            case Heuristics::DIM_3D:
                nextDim = (dim + 1) % 3;
                break;
        }

        NodeID childrenIDs[2];
        childrenIDs[NEAR] = 2 * nodeID + 1; // "Left child" (near)
        childrenIDs[FAR] = 2 * nodeID + 2; // "Right child" (far)

        // Reserve two new tasks indices in the list
        const int childTaskOffset = atomicAdd(outputTasksSize, 2);

        // Add the sum of new ids to the total
        // Also get the current available offset of triangles
        const int currentOutputTriangles = atomicAdd(totalOutputTriangles, outputSizes[NEAR] + outputSizes[FAR]);

        // Compute the children output offsets
        // We multiply by 2 because the global next level output buffer will be twice as large as next level input buffer
        int childOutputMeshOffs[2];
        childOutputMeshOffs[NEAR] = currentOutputTriangles * 2;
        childOutputMeshOffs[FAR] = (currentOutputTriangles * 2) + (outputSizes[NEAR] * 2);

        for(int s = 0; s < 2; s++) // for NEAR and FAR
        {
            const int ti = (childTaskOffset + s);
            outputTasks[ti] = TaskData{
                .dim = nextDim,
                .nodeID = childrenIDs[s],
                .aabb = aabbs[s],
                .level = level + 1,
                .inputMeshOff = outputsOff[s],
                .inputMeshSize = outputSizes[s],
                .outputMeshOff = childOutputMeshOffs[s],
            };
        }
    }
    else
    {
        // Leaf node
        // Store the final mesh in the leaf node

        // Reserve a sub-array in the buffer
        const int off = atomicAdd(leavesBufferSize, taskData.inputMeshSize);

        // Copy all the inputs to the buffer
        for(int i = 0; i < taskData.inputMeshSize; i++)
        {
            leavesBuffer[off + i] = inputTriangles[taskData.inputMeshOff + i];
        }

        node.mesh = &leavesBuffer[off];
        node.meshSize = taskData.inputMeshSize;

        // DO NOT increment totalOutputTriangles
        // Because this variable is used to compute the next output size,
        // but as this is a leaf there is no child node so no need for output for this node for the next level.
    }
}

namespace
{
    CUDA_BOTH float my_saturate(float x)
    {
#ifdef __CUDA_ARCH__
        return __saturatef(x);
#else
        return std::clamp(x, 0.0f, 1.0f);
#endif
    }
}

CUDA_BOTH Point
KDTreeBase::findClosestPointOnTriangle(const Point& query, const Triangle& triangle)
{
    // https://stackoverflow.com/a/32255438/5110937

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
                    s = my_saturate(-d / a);
                    t = 0.f;
                }
                else
                {
                    s = 0.f;
                    t = my_saturate(-e / c);
                }
            }
            else
            {
                s = 0.f;
                t = my_saturate(-e / c);
            }
        }
        else if(t < 0.f)
        {
            s = my_saturate(-d / a);
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
                s = my_saturate(numer / denom);
                t = 1 - s;
            }
            else
            {
                t = my_saturate(-e / c);
                s = 0.f;
            }
        }
        else if(t < 0.f)
        {
            if(a + d > b + e)
            {
                float numer = c + e - b - d;
                float denom = a - 2 * b + c;
                s = my_saturate(numer / denom);
                t = 1 - s;
            }
            else
            {
                s = my_saturate(-e / c);
                t = 0.f;
            }
        }
        else
        {
            float numer = c + e - b - d;
            float denom = a - 2 * b + c;
            s = my_saturate(numer / denom);
            t = 1.f - s;
        }
    }

    return triangle.points[0] + edge0 * s + edge1 * t;
}