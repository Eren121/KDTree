#include "KDTreeDevicePtr.cuh"

KDTreeDevicePtr::KDTreeDevicePtr(const KDTree& tree)
    : m_rootAABB(tree.m_rootAABB),
      m_mesh(tree.m_mesh),
      m_nodes(tree.m_nodes),
      m_leavesBuffer(tree.m_leavesBuffer),
      m_maxLevel(tree.m_heuristics.maxLevel),
      m_totalLeafNodes(tree.m_totalLeafNodes)
{
}

__device__
NPQueryRet KDTreeDevicePtr::findNearestPointOnMesh(const Point& pos) const
{
    NPQueryRet ret{};

    float currentDist = FLT_MAX;

    searchRecursive(pos, currentDist, ret.id, ret.point);

    return ret;
}

__device__
void KDTreeDevicePtr::searchRecursive(const Point& pos,
                                      float& currentDist,
                                      Triangle::ID& currentID,
                                      Point& currentPoint) const
{
    // This should be defined as the maximum level of the array
    constexpr int MAX_STACK_SIZE = 32;
    assert(MAX_STACK_SIZE > m_maxLevel);

    // Recursive may not work well with CUDA
    // We have to do it ourselves
    // Also it is better optimized as we don't pass the references variables because we already have access to them
    struct StackEntry
    {
        NodeID nodeID;
    };
    StackEntry stack[MAX_STACK_SIZE];

    // Pointer to the current entry in the stack
    // We know it is empty when the pointers are equals
    StackEntry *top = stack;

    // Initialize the root stack entry
    // And push to the stack
    // entry is like the end iterator of the stack
    top->nodeID = 0;
    top++;

    // While the stack is not empty
    while(top != stack)
    {
        // Pop from the stack
        top--;

        const NodeID nodeID = top->nodeID;
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

            // Push front entry
            top->nodeID = front;
            top++;
            assert((top - stack) <= MAX_STACK_SIZE);

            // If the current closest point is closer than the closest point of the back face, no need to search in the back
            // face because it will be always further.
            // If so, we save half of the time for the current node
            const float backDist = fabsf(split.p - pos[split.dim]);
            // Do not forget currentDist is squared
            if(backDist * backDist <= currentDist)
            {
                // If it can be closer, search also in this node

                // Push back entry
                top->nodeID = back;
                top++;
                assert((top - stack) <= MAX_STACK_SIZE);
            }
        }
    }
}
