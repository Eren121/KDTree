#pragma once

#include "KDTreeDevicePtr.cuh"

struct QueryOp
{
    KDTreeDevicePtr kd;

    template<typename Tuple>
    __device__
    void operator()(Tuple&& tuple) const
    {
        const Point& query = thrust::get<0>(tuple);
        NPQueryRet& nearest = thrust::get<1>(tuple);

        nearest = kd.findNearestPointOnMesh(query);
    }
};