#include <iostream>
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

int main()
{
    auto points = randomPoints(1'000'000);
    for(int i = 0; i < points.size(); i++) {
        auto& point = points[i];
    }

    auto aabb = KDTree::computeBoundingBox(points);

    KDTree kdtree;

    {
        Timer timer("KDTree build");
        kdtree = KDTree(points);
    }
    {
        const int N = 1'000;
        auto testPts = randomPoints(N, 10.0f);

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

    return 0;
}
