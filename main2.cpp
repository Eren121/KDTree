#include <iostream>

#include <easy3d/core/point_cloud.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/viewer/viewer.h>
#include <easy3d/util/resource.h>
#include <easy3d/util/initializer.h>
#include <easy3d/core/poly_mesh.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/renderer/average_color_blending.h>
#include <easy3d/renderer/dual_depth_peeling.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/core/vec.h>
#include <vector>
#include <random>
#include "KDTree.h"
#include "viewer.h"
#include <chrono>

class Timer {
public:
    Timer(const std::string& title) : title_(title), beg_(clock_::now()) {}
    ~Timer() {
        std::cout << title_ << " elapsed: " << elapsed() << "s" << std::endl;
    }

    void reset() { beg_ = clock_::now(); }

    double elapsed() const {
        return std::chrono::duration_cast<second_> (clock_::now() - beg_).count();
    }

private:
    std::string title_;
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

using namespace easy3d;

std::vector<KDTree::Point> randomPoints(int size, float bounds = 10.0f)
{
    std::vector<KDTree::Point> points;

    std::uniform_real_distribution<float> dist(-bounds, bounds);
    std::mt19937 engine;

    for(int i = 0; i < size; i++)
    {
        KDTree::Point point;

        for(int dim = 0; dim < 3; dim++) {
            point[dim] = dist(engine);
        }

        points.push_back(point);
    }

    return points;
}

vec3 to_easy3d(const KDTree::Point& point)
{
    return {point.x, point.y, point.z};
}

std::unique_ptr<SurfaceMesh> createCube(const KDTree::AABB& aabb)
{
    auto ret = std::make_unique<SurfaceMesh>();

    const float x = aabb.min.x;
    const float y = aabb.min.y;
    const float z = aabb.min.z;
    const float dx = aabb.max.x - aabb.min.x;
    const float dy = aabb.max.y - aabb.min.y;
    const float dz = aabb.max.z - aabb.min.z;

    std::vector<SurfaceMesh::Vertex> v;
    v.push_back(ret->add_vertex({x, y, z}));
    v.push_back(ret->add_vertex({x, y, z+dz}));
    v.push_back(ret->add_vertex({x, y+dy, z}));
    v.push_back(ret->add_vertex({x, y+dy, z+dz}));
    v.push_back(ret->add_vertex({x+dx, y, z}));
    v.push_back(ret->add_vertex({x+dx, y, z+dz}));
    v.push_back(ret->add_vertex({x+dx, y+dy, z}));
    v.push_back(ret->add_vertex({x+dx, y+dy, z+dy}));

    ret->add_triangle(v[0], v[4], v[2]);
    ret->add_triangle(v[4], v[6], v[2]);

    ret->add_triangle(v[7], v[5], v[3]);
    ret->add_triangle(v[5], v[1], v[3]);

    ret->add_triangle(v[0], v[3], v[1]);
    ret->add_triangle(v[2], v[3], v[0]);

    ret->add_triangle(v[4], v[5], v[7]);
    ret->add_triangle(v[4], v[7], v[6]);

    ret->add_triangle(v[0], v[1], v[4]);
    ret->add_triangle(v[1], v[5], v[4]);

    ret->add_triangle(v[2], v[6], v[3]);
    ret->add_triangle(v[6], v[7], v[3]);

    return ret;
}

int main()
{
    initialize();

    // Create a point cloud
    PointCloud cloud;

    auto points = randomPoints(1'000'000);
    for(int i = 0; i < points.size(); i++) {
        auto& point = points[i];
        cloud.add_vertex({point.x, point.y, point.z});
    }

    auto aabb = KDTree::computeBoundingBox(points);

    KDTree kdtree;

    {
        Timer timer("KDTree build");
        kdtree = KDTree(points);
    }
    {
        const int N = 1'000;
        auto testPts = randomPoints(N, 100.0f);

        std::vector<KDTree::Point> resKD(N), resBrute(N);

        {
            Timer timer("KDTree");

            for(int i = 0; i < testPts.size(); i++) {
                resKD[i] = kdtree.computeNearestNeighbor(testPts[i]);
            }
        }
        {
            Timer timer("Bruteforce");

            for(int i = 0; i < testPts.size(); i++) {

                const auto& test = testPts[i];

                // Brute force
                KDTree::Point cur;
                float curDist = FLT_MAX;
                for(const auto& brute : points) {
                    if(brute.distanceSquared(test) < curDist) {
                        curDist = brute.distanceSquared(test);
                        cur = brute;
                    }
                }

                resBrute[i] = cur;
            }
        }

        {
            float delta = 0.0f;
            for(int i = 0; i < N; i++) {
                delta += resKD[i].x - resBrute[i].x;
                delta += resKD[i].y - resBrute[i].y;
                delta += resKD[i].z - resBrute[i].z;
            }

            std::cout << "delta = " << delta << std::endl;
        }
    }


    auto mesh = createCube(aabb);

    MyViewer viewer("Tutorial_504_Transparency");
    viewer.camera()->setViewDirection(vec3(0, 0, -1));
    viewer.camera()->setUpVector(vec3(0, 1, 0));

    viewer.add_model(&cloud, true);
    viewer.add_model(mesh.get(), true);

    {
        auto drawable = mesh->renderer()->get_triangles_drawable("faces");
        drawable->set_opacity(0.5f);
        drawable->set_distinct_back_color(false);
    }
    {
        auto drawable = cloud.renderer()->get_points_drawable("vertices");
        drawable->set_impostor_type(PointsDrawable::SPHERE);
        drawable->set_point_size(10.0f);
        drawable->set_distinct_back_color(false);
    }

    return viewer.run();
}
