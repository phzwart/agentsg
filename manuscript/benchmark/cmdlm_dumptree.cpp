#include <iostream>
#include <set>
#include <vector>
#include <string>
#include "S6.h"
#include "MatS6.h"
#include "B4.h"
#include "C3.h"
#include "G6.h"
#include "P3.h"
#define protected public

#include "LRL_LatticeMatcher.h"
#undef protected

#include "S6.h"
#include "MatS6.h"
int main(int argc,char**argv){
  LRL_LatticeMatcher lm;
  lm.SetRecursionDepth(argc>1?atoi(argv[1]):2);
  S6 ref("-10 -20 -30 -40 -50 -60");
  lm.SetReferenceLattice(ref);
  std::cerr << "tree size " << lm.m_matrixTree.size() << std::endl;
  for (auto& m : lm.m_matrixTree){
    for (size_t i=0;i<36;++i) std::cout << m[i] << (i==35?"\n":" ");
  }
}
