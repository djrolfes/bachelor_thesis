#include "su2Element.hpp"

#ifndef SU2_5CELLElEMENT_HPP
#define SU2_5CELLElEMENT_HPP

#define C5_ETA 0.5590169943749474241022934

class su2_5CellElement : public su2Element
{
public:
  su2_5CellElement()
    : su2Element(){};

  su2_5CellElement(double* el)
    : su2Element(el){

    };

  void renormalize(){};

  su2_5CellElement randomize(double delta, std::mt19937& gen)
  {
    std::uniform_int_distribution<> dist(0, 4);
    return randomize(dist(gen));
  };

protected:
  su2_5CellElement randomize(int direction)
  {
    double vertices[5][4] = { { 1, 0, 0, 0 },
                              { -0.25, C5_ETA, C5_ETA, C5_ETA },
                              { -0.25, -C5_ETA, -C5_ETA, C5_ETA },
                              { -0.25, -C5_ETA, C5_ETA, -C5_ETA },
                              { -0.25, C5_ETA, -C5_ETA, -C5_ETA } };

    return su2_5CellElement(&vertices[direction][0]);
  };
};

#endif /*SU2_5CELLElEMENT_HPP*/
