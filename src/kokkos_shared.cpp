#include <string>
#include "kokkos_shared.h"

DataArray allocate_DataArray(std::string name, int size)
{
  DataArray data;
  for (int i=0; i<NBVAR; ++i)
    data[i] = MyArray(name+std::to_string(i),size);

  return data;

} // allocate_DataArray

DataArrayHost create_mirror(DataArray data)
{

  DataArrayHost datah;
  for (int i=0; i<NBVAR; ++i)
    datah[i] = Kokkos::create_mirror_view(data[i]);

  return datah;

} // create_mirror

