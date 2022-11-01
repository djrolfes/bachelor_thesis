#include "su2Element.hpp"

#ifndef SU2OCTElEMENT_HPP
#define SU2OCTElEMENT_HPP

#define OCT_ONE_OVER_SQRT2 0.7071067811865475244008443
#define OCT_EPS 1e-8

class su2OctElement : public su2Element
{
public:
  su2OctElement()
    : su2Element(){};

  su2OctElement(const su2Element& el)
  {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
  };

  su2OctElement& operator=(const su2Element& el)
  {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
    return *this;
  };

  void renormalize()
  {
    for (int i = 0; i < 4; i++) {
      if (abs(su2Element::element[i]) < OCT_EPS) {
        su2Element::element[i] = 0;
      } else if (roundToOct(&su2Element::element[i], OCT_ONE_OVER_SQRT2)) {
      } else if (roundToOct(&su2Element::element[i], 0.5)) {
      } else if (roundToOct(&su2Element::element[i], 1.0)) {
      } else {
        printf("You just left the Gauge Group [%f,%f,%f,%f]\n",
               su2Element::element[0],
               su2Element::element[1],
               su2Element::element[2],
               su2Element::element[3]);
      }
    }
  };

  su2OctElement randomize(double delta, std::mt19937& gen)
  {
    std::uniform_int_distribution<> dist(0, 5);
    return randomize(dist(gen));
  };

protected:
  su2OctElement randomize(int direction)
  {
    double multEl[4] = { OCT_ONE_OVER_SQRT2, 0, 0, 0 };
    int sign = 1 - (2 * (direction & 1));

    int offset = direction >> 1;

    multEl[1 + offset] = sign * OCT_ONE_OVER_SQRT2;

    return su2OctElement(&multEl[0]) * (*this);
  };

  bool roundToOct(double* el, double roundVal)
  {
    if ((abs(*el) + OCT_EPS) > roundVal && (abs(*el) - OCT_EPS) < roundVal) {
      *el = ((*el > 0) - (*el < 0)) * roundVal;
      return true;
    }
    return false;
  }
};

#endif /*SU2OCTElEMENT_HPP*/
