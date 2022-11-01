#include "su2Element.hpp"

#ifndef SU2_16CELLElEMENT_HPP
#define SU2_16CELLElEMENT_HPP

class su2_16CellElement : public su2Element
{
public:
  su2_16CellElement()
    : su2Element()
  {
    nonZeroIndex = 0;
  };

  su2_16CellElement(double* el, int idx)
    : su2Element(el)
  {
    nonZeroIndex = idx;
  };

  void renormalize(){};

  su2_16CellElement randomize(double delta, std::mt19937& gen)
  {
    std::uniform_int_distribution<> dist(0, 5);
    return randomize(dist(gen));
  };

protected:
  su2_16CellElement randomize(int direction)
  {

    int sign = 1 - (2 * (direction & 1));
    int newNonZeroIndex = (nonZeroIndex + 1 + (direction >> 1)) % 4;

    double newElement[4] = { 0, 0, 0, 0 };
    newElement[newNonZeroIndex] = sign;
    return su2_16CellElement(&newElement[0], sign);
  };
  int nonZeroIndex;
};

#endif /*SU2_16CELLElEMENT_HPP*/
