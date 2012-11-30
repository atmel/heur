#ifndef __VYZKUM_TOPLEVEL__
#define __VYZKUM_TOPLEVEL__

/*We MINIMIZE in whole program!*/

//template<int,typename,int, typename>

//whole program definitions
#define GRID_STEP 0.0001

//error definitions
#include "commons.h"
#include "errors.h"
#include "curandom.h"
#include "range.h"

//build options

#include "abstractInformationProviders.h"
#include "generalMethods.h"
#include "derivedMethod.h"
#include "rangedMasterMethods.h"

//pop include
//#include "candidate.h"
#include "candidateContainer.h"
#include "archive.h"
#include "population.h"

//#include"/stage/initializable.h"

//stages
#include "evaluation.h"
#include "initialization.h"
#include "pertubation.h"
#include "archivationMethod.h"

//#include "merge.h"
//#include "mutation.h"
//#include "selection.h"
//#include "reproduction.h"

//additionals
//#include "probability.h"
//#include "graphics.h"


#endif