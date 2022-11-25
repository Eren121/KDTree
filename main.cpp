#include <iostream>
#include <vector>
#include "KDTree.h"
#include "Utils.h"
#include <cfloat>
#include <omp.h>



int main()
{
    const auto mesh = randomTriangles(50'000);
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
