#include <iostream>
#include <vector>
#include <random>
#include "KDTree.h"
#include <chrono>
#include <cfloat>

class Timer
{
public:
    Timer(const std::string& title) : title_(title), beg_(clock_::now()) {}

    ~Timer()
    {
        std::cout << title_ << " elapsed: " << elapsed() << "s" << std::endl;
    }

    void reset() { beg_ = clock_::now(); }

    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

private:
    std::string title_;
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

std::vector<KDTree::Point> randomPoints(int size, float bounds = 10.0f)
{
    std::vector<KDTree::Point> points;

    std::uniform_real_distribution<float> dist(-bounds, bounds);
    std::mt19937 engine;

    for(int i = 0; i < size; i++)
    {
        KDTree::Point point;

        for(int dim = 0; dim < 3; dim++)
        {
            point[dim] = dist(engine);
        }

        points.push_back(point);
    }

    return points;
}

KDTree::Mesh randomTriangles(int size, float bounds = 10.0f)
{
    KDTree::Mesh ret;

    auto vertices = randomPoints(size * 3, bounds);

    for(int i = 0; i < vertices.size() / 3; i++)
    {
        ret.emplace_back(
            vertices[i * 3 + 0],
            vertices[i * 3 + 1],
            vertices[i * 3 + 2]
        );
    }

    return ret;
}

int main()
{
    const auto mesh = randomTriangles(500'000);
    const auto aabb = KDTree::computeBoundingBox(mesh);

    KDTree kdtree;

    {
        Timer timer("KDTree build");
        kdtree = KDTree(mesh);
    }
    {
        const int N = 100;
        auto testPts = randomPoints(N, 10.0f);

        std::vector<KDTree::Point> resKD(N), resBrute(N);

        {
            Timer timer("KDTree");

            for(int i = 0; i < testPts.size(); i++)
            {
                auto ret = kdtree.findNearestPointOnMesh(testPts[i]);
                resKD[i] = ret.point;
            }
        }
        {
            Timer timer("Bruteforce");

            for(int i = 0; i < testPts.size(); i++)
            {

                const auto& test = testPts[i];

                float curDist = FLT_MAX;
                for(int id = 0; id < mesh.size(); id++)
                {
                    const auto& triangle = mesh[id];

                    const auto nearest = KDTree::findClosestPointOnTriangle(test, triangle);
                    if(nearest.distanceSquared(test) < curDist)
                    {
                        curDist = nearest.distanceSquared(test);
                        resBrute[i] = nearest;
                    }
                }
            }
        }

        {
            float delta = 0.0f;
            for(int i = 0; i < N; i++)
            {
                delta += resKD[i].x - resBrute[i].x;
                delta += resKD[i].y - resBrute[i].y;
                delta += resKD[i].z - resBrute[i].z;
            }

            std::cout << "delta = " << delta << std::endl;
        }
    }

    return 0;
}
