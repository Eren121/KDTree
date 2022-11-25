#include <GU/GU_Detail.h>
#include <GU/GU_PrimPolySoup.h>
#include <GU/GU_DetailInfo.h>
#include <UT/UT_Exit.h>
#include <GU/GU_RayIntersect.h>
#include <cstddef>
#include <iostream>
#include "KDTree.h"
#include "Utils.h"

long getRayIntersectPrimitive(GU_RayIntersect& ray, UT_Vector3F p, float& u, float& v, float& w)
{
    long polyId;

    GU_MinInfo mininfo;

    mininfo.init();
    ray.minimumPoint(p, mininfo);

    // We could not find the primitive intersection
    if(!mininfo.prim)
    {
        std::cout << "error" << std::endl;
        return -1;
    }

    //Get the Primitive ID
    const GEO_Primitive *geoPrim = mininfo.prim;
    polyId = geoPrim->getMapOffset();

    //se uv values of the rayIntersect point
    u = mininfo.u1;
    v = mininfo.v1;
    w = mininfo.w1;

    // To determine the actual 3D coordinates of the closest point, the UV coords have to be evaluated by the primitive:
    //Solution from:
    //https://forums.odforce.net/topic/8335-hdk-spatial-geometry-storage-solved/
    UT_Vector4 result;
    if(mininfo.prim)
    {
        mininfo.prim->evaluateInteriorPoint(result, mininfo.u1, mininfo.v1);
        u = result.x();
        v = result.y();
        w = result.z();

        //mininfo.prim-&gt;evaluateInteriorPoint(result, mininfo.u, mininfo.v);
        //The 3D point is now stored in "result"!
        //Now you can add it to the GDP or do whatever!
    }
    else
    {
        std::cout << "error" << std::endl;
    }


    return polyId;
}

int main(int argc, char *argv[])
{
    std::string home = std::getenv("HOME");

    //const std::string path = home + "/Documents/LSTS/DamBreak/dambreak_fluidSim_SlowTImescal0_25_100.bgeo";
    const std::string path = home + "/models/res1.bgeo";

    GU_Detail gdp;

    const auto status = gdp.load(path.c_str());
    if(!status.success())
    {
        std::cout << "Error loading bgeo file" << std::endl;
        exit(1);
    }

    std::unique_ptr<GU_RayIntersect> ray;

    {
        Timer("GU_RayIntersect build");
        ray = std::make_unique<GU_RayIntersect>(&gdp);
    }

    KDTree kdtree;

    {
        KDTree::Mesh mesh;

        // Bump the vertices from Houdini
        // Get the position
        GEO_Primitive *prim;
        GA_FOR_ALL_PRIMITIVES(&gdp, prim)
        {
            KDTree::Triangle tri;
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    tri.points[i][j] = prim->getPos3(i)(j);
                }
            }

            mesh.push_back(tri);
        }

        {
            std::cout << mesh.size() << " triangles" << std::endl;
            Timer timer("KDTree build");
            kdtree = KDTree(std::move(mesh));
        }

    }

    UT_BoundingBox bounds;
    gdp.getCachedBounds(bounds);

    {
        const int N = 10'000;
        auto testPts = randomPoints(N, 0.1f);

        for(int i = 0; i < testPts.size(); i++)
        {
            float x = rand() / float(RAND_MAX) * (bounds.xmax() - bounds.xmin()) + bounds.xmin();
            float y = rand() / float(RAND_MAX) * (bounds.ymax() - bounds.ymin()) + bounds.ymin();
            float z =rand() / float(RAND_MAX) * (bounds.zmax() - bounds.zmin()) + bounds.zmin();

            testPts[i] = {x, y, z};
        }

        std::vector<KDTree::Point> resKD(N), resH(N);

        {
            Timer timer("KDTree");

            for(int i = 0; i < testPts.size(); i++)
            {
                auto ret = kdtree.findNearestPointOnMesh(testPts[i]);
                resKD[i] = ret.point;
            }
        }
        {
            Timer timer("Houdini");

            for(int i = 0; i < testPts.size(); i++)
            {

                const auto& test = testPts[i];

                getRayIntersectPrimitive(*ray, {test[0], test[1], test[2]}, resH[i].x, resH[i].y, resH[i].z);
            }
        }

        {
            float delta = 0.0f;
            for(int i = 0; i < N; i++)
            {
                delta += std::abs(resKD[i].x - resH[i].x);
                delta += std::abs(resKD[i].y - resH[i].y);
                delta += std::abs(resKD[i].z - resH[i].z);
            }

            std::cout << "delta = " << delta << std::endl;
        }
    }

    UT_Exit::exit(UT_Exit::EXIT_OK); // exit with proper tear down
}