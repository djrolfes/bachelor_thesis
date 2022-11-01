#include "su2Element.hpp"

#ifndef SU2LISTELEMENT_HPP
#define SU2LISTELEMENT_HPP

struct listElementData
{
  su2Element element;
  int* neighborList;
  int neighborCount;
  double weight;
};

class su2ListElement : public su2Element
{
public:
  su2ListElement(listElementData* iElementList)
  {

    elementList = iElementList;
    index = 0;

    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = elementList[index].element[i];
    }
  };

  su2ListElement(int iIndex, listElementData* iElementList)
  {

    elementList = iElementList;
    index = iIndex;

    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = elementList[index].element[i];
    }
  };

  void renormalize(){};

  su2ListElement randomize(double delta, std::mt19937& gen)
  {
    std::uniform_int_distribution<> dist(0,
                                         elementList[index].neighborCount - 1);

    int n = dist(gen);
    return randomize(n);
  };

  double getWeight() { return elementList[index].weight; };

private:
  su2ListElement randomize(int newIdx)
  {
    return su2ListElement(elementList[index].neighborList[newIdx], elementList);
  };

  listElementData* elementList;
  int index;
};

#endif /*SU2ELEMENT_HPP*/
