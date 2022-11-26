#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "KDTreeDevicePtr.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "QueryOp.h"

std::vector<Triangle> simple2DMesh()
{
    return {
        {
            {-4, -3, 0},
            {-3, -3 ,0},
            {-4, -2, 0}
        },
        {
            {-1, -2, 0},
            {1, -2, 0},
            {-1, -1, 0}
        },
        {
            {-4, 3, 0},
            {-3, 3, 0},
            {-4, 4, 0}
        },
        {
            {3, 3, 0},
            {4, 3, 0},
            {3, 4, 0}
        }
    };
}

TEST_CASE("Simple KD-Tree 2D", "[kd]")
{
    KDTree::Heuristics heu;
    heu.dim = KDTree::Heuristics::DIM_2D;
    heu.maxNodeSize = 1;

    KDTree kd(simple2DMesh(), heu);

    const std::vector<Point> query = {
        {2, -2, 0},
        {-3, -2, 0}
    };
    const std::vector<Point> expected{
        {1, -2, 0},
        {-3.5f, -2.5f, 0}
    };

    thrust::device_vector<Point> d_query(query);
    thrust::device_vector<NPQueryRet> d_res(d_query.size());


    QueryOp op{KDTreeDevicePtr(kd)};

    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_query.begin(), d_res.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(d_query.end(), d_res.end()));
    thrust::for_each(first, last, op);

    thrust::host_vector<NPQueryRet> res = d_res;

    for(int i = 0; i < res.size(); i++)
    {
        const float tolerance = 0.001f;
        REQUIRE_THAT(res[i].point.x, Catch::Matchers::WithinRel(expected[i].x, tolerance));
        REQUIRE_THAT(res[i].point.y, Catch::Matchers::WithinRel(expected[i].y, tolerance));
        REQUIRE_THAT(res[i].point.z, Catch::Matchers::WithinRel(expected[i].z, tolerance));
    }
}