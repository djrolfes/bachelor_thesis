#include "partitions.hpp"
#include "rapidcsv.h"
#include "su2Element.hpp"

#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetSet.h"
#include "libqhullcpp/QhullVertex.h"

#include <vector>

#ifndef DISCRETIZER_HPP
#define DISCRETIZER_HPP

class discretizer
{
public:
  discretizer(std::string fileName)
  {
    rapidcsv::Document doc(
      fileName, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams('\t'));

    // Read elements from file

    N = doc.GetRowCount();
    elementData = new listElementData[N];

    for (int i = 0; i < N; i++) {
      double el[4];
      for (int j = 0; j < 4; j++) {
        el[j] = doc.GetCell<double>(j, i);
      }
      elementData[i].element = su2Element(&el[0]);
    }

    if (doc.GetColumnCount() == 5) {
      for (int i = 0; i < N; i++) {
        elementData[i].weight = doc.GetCell<double>(4, i);
      }
    } else {
      for (int i = 0; i < N; i++) {
        elementData[i].weight = 1.0;
      }
    }

    findNeighbors();
  };

  ~discretizer()
  {

    for (int i = 0; i < N; i++) {
      delete[] elementData[i].neighborList;
    }

    delete[] elementData;
  };

  void findNeighbors()
  {

    double points[4 * N];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < 4; j++) {
        points[(4 * i) + j] = elementData[i].element[j];
      }
    }

    orgQhull::Qhull q("", 4, N, &points[0], "");
    q.defineVertexNeighborFacets();

    for (orgQhull::QhullVertexList::ConstIterator i = q.vertexList().begin();
         i != q.vertexList().end();
         ++i) {

      std::vector<int> neighborBuffer;

      for (orgQhull::QhullVertexList::ConstIterator j = q.vertexList().begin();
           j != q.vertexList().end();
           ++j) {
        if (i->id() != j->id()) {

          for (orgQhull::QhullFacet neighbor : i->neighborFacets()) {
            if (j->neighborFacets().contains(neighbor)) {
              neighborBuffer.push_back(j->point().id());
              break;
            }
          }
        }
      }

      elementData[i->point().id()].neighborCount = neighborBuffer.size();
      elementData[i->point().id()].neighborList =
        new int[neighborBuffer.size()];

      for (int j = 0; j < neighborBuffer.size(); j++) {
        elementData[i->point().id()].neighborList[j] = neighborBuffer[j];
      }
    }
  };

  listElementData* getElementData() { return elementData; };
  int getElementCount() { return N; };

private:
  int N;
  listElementData* elementData;
};

#endif /*DISCRETIZER_HPP*/
