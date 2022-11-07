#include "su2Element.hpp"

#ifndef Z2ELEMENT_HPP
#define Z2ELEMENT_HPP

class z2Element : public su2Element
{
public:
  z2Element()
    : z2Element(1){};

  z2Element(int iState)
  {
    state = iState;
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = 0;
    }
    su2Element::element[0] = state;
  };

  void renormalize(){};

  z2Element randomize(double delta, std::mt19937& gen)
  {
    return randomize(std::uniform_real_distribution<double>(0., 1.)(gen));
  };

protected:
  z2Element randomize(double rand)
  {
    if (rand > .5) {
      return z2Element(-1 * state);
    } else {
      return z2Element(state);
    }
  };
  int state;
};

#endif /*Z2ELEMENT_HPP*/
