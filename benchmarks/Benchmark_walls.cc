#include <Grid.h>

int main (int argc, char ** argv)
{
  // Initialize grid library
  Grid::Grid_init(&argc,&argv);

  // Scalar field in Grid
  typedef Grid::iScalar<Grid::vReal> vScalar;
  typedef Grid::Lattice<vScalar>     vScalarField;

  int Nloop = 100000; // Benchmark iterations
  int Nd    = 3;      // Three spatial dimensions

  // SIMD and MPI layouts
  std::vector<int> simd_layout = Grid::GridDefaultSimd(Nd,Grid::vReal::Nsimd());
  std::vector<int> mpi_layout  = std::vector<int>(Nd,1);

  // OpenMP thread count
  int threads = Grid::GridThread::GetThreads();
  std::cout << Grid::GridLogMessage << "Grid is setup to use "<<threads<<" threads"<<std::endl;

  // Standard output header
  std::cout << Grid::GridLogMessage << "====================================================================================================" <<std::endl;
  std::cout << Grid::GridLogMessage << "= Benchmarking Walls 3D" << std::endl;
  std::cout << Grid::GridLogMessage << "====================================================================================================" <<std::endl;
  std::cout << Grid::GridLogMessage << "  L  " << "\t\t" << "bytes" << "\t\t\t" << "GB/s" << "\t\t" << "Gflop/s" << std::endl;
  std::cout << Grid::GridLogMessage << "----------------------------------------------------------" << std::endl;

  // Physical constants

  const Grid::Real alfa       = 3.0;
  const Grid::Real beta0      = 0.0;
  const Grid::Real W0         = 10.0;

  const Grid::Real expoente   = 0.6666666;
  const Grid::Real ctinitial  = 1.0;
  const Grid::Real tampasso   = 0.25;

  const Grid::Real dct        = tampasso * ctinitial;
  const Grid::Real expoentec  = expoente / (1.0 - expoente);
  const Grid::Real dx2r       = 1.0 / (ctinitial * ctinitial);
  const Grid::Real ct         = ctinitial;
  const Grid::Real omega0     = 0.5 * W0 * ctinitial;
  const Grid::Real V0         = 0.5 * M_PI * M_PI / ( omega0 * omega0);

  const Grid::Real delta      = 0.5 * alfa * expoentec * dct / ct;
  const Grid::Real idct       = 1.0 / dct;
  const Grid::Real m1delta    = 1.0 - delta;
  const Grid::Real V04atbeta0 = 4.0 * V0 * pow( pow(ctinitial,expoentec), beta0 );
  const Grid::Real qmiu       = 0.0;
  const Grid::Real i1pdelta   = 1.0 / (1.0 + delta);

  // Iterate over different local grid sizes given by lat.lat.lat
  for(int lat=4; lat<=32; lat+=4)
  {
    // Lattice size and volume
    std::vector<int> latt_size ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2]});
    int vol = latt_size[0]*latt_size[1]*latt_size[2];

    // Cartesian grid
    Grid::GridCartesian Grid(latt_size,simd_layout,mpi_layout);

    // Random number genereator
    Grid::GridParallelRNG pRNG(&Grid);
    pRNG.SeedRandomDevice();

    // Scalar fields
    vScalarField P1(&Grid);
    vScalarField P2(&Grid);
    vScalarField * Pold;
    vScalarField * Pnow;
    vScalarField * Pnew;

    // Initialise scalar fields
    random(pRNG,P1);
    P2 = Grid::zero;
    Pold = NULL;
    Pnow = &P1;
    Pnew = &P2;

    // Cartesian stencil
    int npoint = 6;
    int Even   = 0;
    const std::vector<int> directions   ({ 0, 1, 2, 0, 1, 2});
    const std::vector<int> displacements({ 1, 1, 1,-1,-1,-1});
    Grid::CartesianStencil stencil(&Grid,npoint,Even,directions,displacements);

    // Check that P1 and P2 are consistent
    Grid::conformable(P1._grid,P2._grid);
    P2.checkerboard = P1.checkerboard;

    // Communications buffer
    std::vector< vScalar, Grid::alignedAllocator<vScalar> > comm_buf;
    comm_buf.resize(stencil._unified_buffer_size);

    // Compressor
    Grid::SimpleCompressor<vScalar> compressor;

    // Initialise Pnew

    // Halo exchange
    stencil.HaloExchange(P1,comm_buf,compressor);

    // Start timer
    double start = Grid::usecond();

    // Iterate over all sites on the grid
    for(int i=0; i<Pnow->_grid->oSites(); i++)
    {
      // Variables used by stencil
      Grid::StencilEntry *SE;
      vScalar SV;
      vScalar Lphi;
      int permute_type;

      // Calculate Laplacian in 3 dimensions from 6-point stencil

      // Stencil entry +x
      SE = stencil.GetEntry(permute_type,0,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi = SV;

      // Stencil entry +y
      SE = stencil.GetEntry(permute_type,1,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi += SV;

      // Stencil entry +z
      SE = stencil.GetEntry(permute_type,2,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi += SV;

      // Stencil entry -x
      SE = stencil.GetEntry(permute_type,3,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi += SV;

      // Stencil entry -y
      SE = stencil.GetEntry(permute_type,4,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi += SV;

      // Stencil entry -z
      SE = stencil.GetEntry(permute_type,5,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      Lphi += SV;

      // Complete Laplacian
      Lphi -= 6.0 * Pnow->_odata[i];
      Lphi *= dx2r;

      // Derivatives
      vScalar DPnow_ijkl = Grid::zero;
      vScalar DPnew_ijkl = (m1delta * DPnow_ijkl +
              dct * (Lphi - V04atbeta0 *((Pnow->_odata[i]*(Pnow->_odata[i]*Pnow->_odata[i]-1.0)) + qmiu))) * i1pdelta;

      // Dynamical evolution of P
      Pnew->_odata[i] = Pnow->_odata[i] + dct * DPnew_ijkl;
    }

    // Total run time
    double stop = Grid::usecond();
    double time = (stop-start) * 1.0E-6;

    // Memory throughput and Flop rate
    double bytes = vol * sizeof(Grid::Real) * 2;
    double flops = vol * 20;
    std::cout << Grid::GridLogMessage << std::setprecision(3) << lat
              << "\t\t" << bytes << " \t\t" << 1.0E-9 * bytes / time
              << "\t\t" << 1.0E-9 * flops / time << std::endl;
  }

  Grid::Grid_finalize();
  return(0);
}
