#include "metropolizer.hpp"

#include "executor.hpp"
#include "partitions.hpp"
#include <random>

template<int dim, class su2Type>
void
initFieldType(su2Type* fields, int nMax, bool cold, int iterations = 1)
{
  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2Type();
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < iterations; i++) {
          fields[loc] = fields[loc].randomize(1.0, generator);
        }
      }
    }
  }
}

template<int dim, class su2Type>
void
initVolleyFields(su2Type* fields, int nMax, bool cold, int subdivs)
{

  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2Type(subdivs);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < (subdivs + 2) * 200; i++) {
          fields[loc] = fields[loc].randomize(1.0, generator);
        }
      }
    }
  }
}

template<int dim>
void
initListFields(su2ListElement* fields,
               int nMax,
               bool cold,
               listElementData* elementList,
               int iterations = 1)
{
  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2ListElement(0, elementList);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < iterations; i++) {
          fields[loc] = fields[loc].randomize(1.0, generator);
        }
      }
    }
  }
}

template<int dim>
executor<dim>::executor(int iLatSize,
                        double iBeta,
                        int iMultiProbe,
                        double iDelta,
                        int iPartType,
                        std::string iPartFile,
                        int iSubdivs)
  : action(iLatSize, iBeta)
{
  multiProbe = iMultiProbe;
  delta = iDelta;
  partType = iPartType;
  partFile = iPartFile;
  subdivs = iSubdivs;

  int fieldsSize = dim * action.getSiteCount();
  switch (partType) {
    case SU2_ELEMENT:
      fieldsSize *= sizeof(su2Element);
      break;
    case SU2_TET_ELEMENT:
      fieldsSize *= sizeof(su2TetElement);
      break;
    case SU2_OCT_ELEMENT:
      fieldsSize *= sizeof(su2OctElement);
      break;
    case SU2_ICO_ELEMENT:
      fieldsSize *= sizeof(su2IcoElement);
      break;
    case SU2_LIST_ELEMENT:
      fieldsSize *= sizeof(su2ListElement);
      break;
    case SU2_VOLLEY_ELEMENT:
      fieldsSize *= sizeof(su2VolleyElement<false>);
      break;
    case SU2_WEIGHTED_VOLLEY_ELEMENT:
      fieldsSize *= sizeof(su2VolleyElement<true>);
      break;
    case SU2_LINEAR_ELEMENT:
      fieldsSize *= sizeof(su2LinearElement<false>);
      break;
    case SU2_WEIGHTED_LINEAR_ELEMENT:
      fieldsSize *= sizeof(su2LinearElement<true>);
      break;
    case SU2_5_CELL_ELEMENT:
      fieldsSize *= sizeof(su2_5CellElement);
      break;
    case SU2_16_CELL_ELEMENT:
      fieldsSize *= sizeof(su2_16CellElement);
      break;
    case SU2_120_CELL_ELEMENT:
      fieldsSize *= sizeof(su2_120CellElement);
      break;
    case Z2_ELEMENT:
      fieldsSize *= sizeof(z2Element);
      break;
  }

  fields = malloc(fieldsSize);
}
template<int dim>
executor<dim>::~executor()
{
  free(fields);

  if (partType == SU2_LIST_ELEMENT) {
    delete disc;
  }
}

template<int dim>
void
executor<dim>::initFields(bool cold)
{

  switch (partType) {
    case SU2_ELEMENT:
      initFieldType<dim, su2Element>(
        (su2Element*)fields, action.getSiteCount(), cold);
      break;
    case SU2_TET_ELEMENT:
      initFieldType<dim, su2TetElement>(
        (su2TetElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_OCT_ELEMENT:
      initFieldType<dim, su2OctElement>(
        (su2OctElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_ICO_ELEMENT:
      initFieldType<dim, su2IcoElement>(
        (su2IcoElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_LIST_ELEMENT:
      loadListFields(cold);
      break;
    case SU2_VOLLEY_ELEMENT:
      initVolleyFields<dim, su2VolleyElement<false>>(
        (su2VolleyElement<false>*)fields, action.getSiteCount(), cold, subdivs);
      break;
    case SU2_WEIGHTED_VOLLEY_ELEMENT:
      initVolleyFields<dim, su2VolleyElement<true>>(
        (su2VolleyElement<true>*)fields, action.getSiteCount(), cold, subdivs);
      break;
    case SU2_LINEAR_ELEMENT:
      initVolleyFields<dim, su2LinearElement<false>>(
        (su2LinearElement<false>*)fields, action.getSiteCount(), cold, subdivs);
      break;
    case SU2_WEIGHTED_LINEAR_ELEMENT:
      initVolleyFields<dim, su2LinearElement<true>>(
        (su2LinearElement<true>*)fields, action.getSiteCount(), cold, subdivs);
      break;
    case SU2_5_CELL_ELEMENT:
      initFieldType<dim, su2_5CellElement>(
        (su2_5CellElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_16_CELL_ELEMENT:
      initFieldType<dim, su2_16CellElement>(
        (su2_16CellElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_120_CELL_ELEMENT:
      initFieldType<dim, su2_120CellElement>(
        (su2_120CellElement*)fields, action.getSiteCount(), cold, 500);
      break;
    case Z2_ELEMENT:
      initFieldType<dim, z2Element>(
        (z2Element*)fields, action.getSiteCount(), cold, 500);
      break;
  }
}

template<int dim>
void
executor<dim>::run(int measurements, int multiSweep, std::string outFile)
{
  std::ofstream file;
  file.open(outFile);

  switch (partType) {
    case SU2_ELEMENT:
      this->runMetropolis<su2Element>(measurements, multiSweep, file);
      break;
    case SU2_TET_ELEMENT:
      this->runMetropolis<su2TetElement>(measurements, multiSweep, file);
      break;
    case SU2_OCT_ELEMENT:
      this->runMetropolis<su2OctElement>(measurements, multiSweep, file);
      break;
    case SU2_ICO_ELEMENT:
      this->runMetropolis<su2IcoElement>(measurements, multiSweep, file);
      break;
    case SU2_LIST_ELEMENT:
      this->runMetropolis<su2ListElement>(measurements, multiSweep, file);
      break;
    case SU2_VOLLEY_ELEMENT:
      this->runMetropolis<su2VolleyElement<false>>(
        measurements, multiSweep, file);
      break;
    case SU2_WEIGHTED_VOLLEY_ELEMENT:
      this->runMetropolis<su2VolleyElement<true>>(
        measurements, multiSweep, file);
      break;
    case SU2_LINEAR_ELEMENT:
      this->runMetropolis<su2LinearElement<false>>(
        measurements, multiSweep, file);
      break;
    case SU2_WEIGHTED_LINEAR_ELEMENT:
      this->runMetropolis<su2LinearElement<true>>(
        measurements, multiSweep, file);
      break;
    case SU2_5_CELL_ELEMENT:
      this->runMetropolis<su2_5CellElement>(measurements, multiSweep, file);
      break;
    case SU2_16_CELL_ELEMENT:
      this->runMetropolis<su2_16CellElement>(measurements, multiSweep, file);
      break;
    case SU2_120_CELL_ELEMENT:
      this->runMetropolis<su2_120CellElement>(measurements, multiSweep, file);
      break;
    case Z2_ELEMENT:
      this->runMetropolis<z2Element>(measurements, multiSweep, file);
      break;
  }

  file.close();
}

template<int dim>
void
executor<dim>::loadListFields(bool cold)
{
  disc = new discretizer(partFile);

  initListFields<dim>((su2ListElement*)fields,
                      action.getSiteCount(),
                      cold,
                      disc->getElementData(),
                      disc->getElementCount() * 100);
}

template<int dim>
template<class su2Type>
void
executor<dim>::runMetropolis(int measurements,
                             int multiSweep,
                             std::ofstream& outFile)
{

  metropolizer<dim, su2Type> metro(action, multiProbe, delta, (su2Type*)fields);
  for (int i = 0; i < measurements; i++) {
    double plaquette = metro.sweep(multiSweep);
    this->logResults(i, plaquette, metro.getHitRate(), outFile);
  }
}

template<int dim>
void
executor<dim>::logResults(int i,
                          double plaquette,
                          double hitRate,
                          std::ofstream& file)
{
  std::cout << i << " " << std::scientific << std::setw(18)
            << std::setprecision(15) << plaquette << " " << hitRate
            << std::endl;
  file << i << "\t" << std::scientific << std::setw(18) << std::setprecision(15)
       << plaquette << "\t" << hitRate << std::endl;
}
