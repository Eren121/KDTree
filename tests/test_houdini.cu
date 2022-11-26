#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "KDTreeDevicePtr.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "houdini.hpp"
#include "Utils.h"
#include "QueryOp.h"

TEST_CASE("Houdini with Mesh")
{
    hou::Mesh mesh(MODELS_DIR "/bun_zipper.ply");
    auto queries = randomPoints(100'000, 0.1f);
    std::cout << "Load mesh with " << mesh.triangles.size() << "triangles" << std::endl;
    auto expected = mesh.query_np(queries);

    thrust::device_vector<Point> d_query(queries);
    thrust::device_vector<NPQueryRet> d_res(d_query.size());

    KDTree kd;

    {
        KDTree::Heuristics heu;
        heu.maxLevel = 10;

        Timer t("Build KDTree");
        kd = KDTree(mesh.triangles, heu);
    }

    QueryOp op{KDTreeDevicePtr(kd)};

    {
        Timer t("Queries KDTree");
        auto first = thrust::make_zip_iterator(thrust::make_tuple(d_query.begin(), d_res.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(d_query.end(), d_res.end()));
        thrust::for_each(first, last, op);
    }

    thrust::host_vector<NPQueryRet> res = d_res;

    for(int i = 0; i < res.size(); i++)
    {
        const float tolerance = 0.00001f;
        REQUIRE_THAT(res[i].point.x, Catch::Matchers::WithinRel(expected[i].point.x, tolerance));
        REQUIRE_THAT(res[i].point.y, Catch::Matchers::WithinRel(expected[i].point.y, tolerance));
        REQUIRE_THAT(res[i].point.z, Catch::Matchers::WithinRel(expected[i].point.z, tolerance));
    }
}