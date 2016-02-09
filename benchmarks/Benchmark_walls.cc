#include <Grid.h>

int main (int argc, char ** argv)
{
  // Initialize grid library
  Grid::Grid_init(&argc,&argv);

  // Scalar field in Grid
  typedef Grid::iScalar<Grid::vReal> vScalar;
  typedef Grid::Lattice<vScalar>     vScalarField;

  int Nloop = 100000; // Benchmark iterations
  int Nd    = 3;    // Three spatial dimensions

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

    // Scalar fields P1 and Lphi
    vScalarField P1(&Grid);
    vScalarField Lphi(&Grid);

    // Initialise P1 (random) and Lphi (zeros)
    random(pRNG,P1);
    Lphi = Grid::zero;

    // Cartesian stencil
    int npoint = 6;
    int Even   = 0;
    const std::vector<int> directions   ({ 0, 1, 2, 0, 1, 2});
    const std::vector<int> displacements({ 1, 1, 1,-1,-1,-1});
    Grid::CartesianStencil stencil(&Grid,npoint,Even,directions,displacements);

    // Check that P1 and Lphi are consistent
    Grid::conformable(P1._grid,Lphi._grid);
    Lphi.checkerboard = P1.checkerboard;

    // Communications buffer
    std::vector< vScalar, Grid::alignedAllocator<vScalar> > comm_buf;
    comm_buf.resize(stencil._unified_buffer_size);

    // Compressor
    Grid::SimpleCompressor<vScalar> compressor;

    // Start timer
    double start = Grid::usecond();

    // Iterate Nloop times for benchmark
    for(int i=0;i<Nloop;i++)
    {
      // Halo exchange
      stencil.HaloExchange(P1,comm_buf,compressor);

      // Iterate over all sites on the grid
      for(int i=0;i<Lphi._grid->oSites();i++){

        // Variables used by stencil
        Grid::StencilEntry *SE;
        vScalar SV;
        int permute_type;

        // Calculate Laplacian in 3 dimensions from 6-point stencil

        // Stencil entry +x
        SE = stencil.GetEntry(permute_type,0,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] = SV;

        // Stencil entry +y
        SE = stencil.GetEntry(permute_type,1,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] += SV;

        // Stencil entry +z
        SE = stencil.GetEntry(permute_type,2,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] += SV;

        // Stencil entry -x
        SE = stencil.GetEntry(permute_type,3,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] += SV;

        // Stencil entry -y
        SE = stencil.GetEntry(permute_type,4,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] += SV;

        // Stencil entry -z
        SE = stencil.GetEntry(permute_type,5,i);
        if ( SE->_is_local && SE->_permute )
          permute(SV,P1._odata[SE->_offset],permute_type);
        else if (SE->_is_local)
          SV = P1._odata[SE->_offset];
        else
          SV = comm_buf[SE->_offset];
        Lphi._odata[i] += SV;

        // Subtract central term
        Lphi._odata[i] -= 6.0 * P1._odata[i];
      }
    }

    // Total run time
    double stop = Grid::usecond();
    double time = (stop-start) * 1.0E-6 / Nloop;

    // Memory throughput and Flop rate
    double bytes = vol * sizeof(Grid::Real) * 2;
    double flops = vol * 7;
    std::cout << Grid::GridLogMessage << std::setprecision(3) << lat
              << "\t\t" << bytes << " \t\t" << 1.0E-9 * bytes / time
              << "\t\t" << 1.0E-9 * flops / time << std::endl;
  }

  Grid::Grid_finalize();
  return(0);
}
