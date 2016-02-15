#include <Grid.h>

class Walls
{

public:

  typedef Grid::iScalar<Grid::vReal> vScalar;
  typedef Grid::Lattice<vScalar>     vScalarField;

  Walls(Grid::GridCartesian * Grid) :

    stencil(Grid,npoint,Even,directions,displacements),
    dct(tampasso * ctinitial),
    expoentec(expoente / (1.0 - expoente)),
    dx2r(1.0 / (ctinitial * ctinitial)),
    omega0(0.5 * W0 * ctinitial),
    V0(0.5 * M_PI * M_PI / ( omega0 * omega0)),
    idct(1.0 / dct),
    qmiu(0.0)

  {
    // Resize communication buffer
    comm_buf.resize(stencil._unified_buffer_size);
  }

  void initialise(Walls::vScalarField * Pnew, const Walls::vScalarField * Pnow, const Walls::vScalarField * Pold)
  {
    // Initialise global varaibles
    ct         = ctinitial;
    delta      = 0.5 * alfa * expoentec * dct / ct;
    V04atbeta0 = 4.0 * V0 * pow( pow(ctinitial,expoentec), beta0 );
    m1delta    = 1.0 - delta;
    i1pdelta   = 1.0 / (1.0 + delta);

    // Halo exchange
    stencil.HaloExchange(*Pnow,comm_buf,compressor);

    // Iterate over all grid sites
    PARALLEL_FOR_LOOP
    for(int i=0; i<Pnow->_grid->oSites(); i++)
    {
      // Laplacian
      vScalar Lphi = Laplacian(Pnow,stencil,comm_buf,i);
      // Current time derivative
      vScalar DPnow_ijkl = Grid::zero;
      // New time derivative
      vScalar DPnew_ijkl = (m1delta * DPnow_ijkl + dct * (Lphi - V04atbeta0 *((Pnow->_odata[i]*(Pnow->_odata[i]*Pnow->_odata[i]-1.0)) + qmiu))) * i1pdelta;
      // Dynamical evolution of P
      Pnew->_odata[i] = Pnow->_odata[i] + dct * DPnew_ijkl;
    }

    // Halo exchange
    stencil.HaloExchange(*Pnew,comm_buf,compressor);

    // Increment time
    ct_prev = ct;
    ct += dct;
  }

  void timestep(Walls::vScalarField * Pnew, const Walls::vScalarField * Pnow, const Walls::vScalarField * Pold)
  {
    // Calculate global varaibles
    delta      = 0.5 * alfa * expoentec * dct / ct;
    i1pdelta   = 1.0 / (1.0 + delta);
    m1delta    = 1.0 - delta;
    V04atbeta0 = 4.0 * V0 * pow( pow(ct_prev,expoentec), beta0 );

    // Iterate over all grid sites
    PARALLEL_FOR_LOOP
    for(int i=0; i<Pnow->_grid->oSites(); i++)
    {
      // Laplacian
      vScalar Lphi = Laplacian(Pnow,stencil,comm_buf,i);
      // Current time derivative
      vScalar DPnow_ijkl = (Pnow->_odata[i] - Pold->_odata[i]) * idct;
      // New time derivative
      vScalar DPnew_ijkl = (m1delta * DPnow_ijkl + dct * (Lphi - V04atbeta0 * ((Pnow->_odata[i]*(Pnow->_odata[i]*Pnow->_odata[i]-1.0)) + qmiu))) * i1pdelta;
      // Dynamical evolution of P
      Pnew->_odata[i] = Pnow->_odata[i] + dct * DPnew_ijkl;
    }

    // Halo exchange
    stencil.HaloExchange(*Pnew,comm_buf,compressor);

    // Increment time
    ct_prev = ct;
    ct += dct;
  }

  Walls::vScalar Laplacian(const Walls::vScalarField * Pnow, Grid::CartesianStencil & stencil, std::vector< Walls::vScalar, Grid::alignedAllocator<Walls::vScalar> > & comm_buf, int i)
  {
    // Variables used by stencil
    Grid::StencilEntry *SE;
    vScalar SV;
    int permute_type;

    // Return value
    vScalar Lphi;

    // Calculate Laplacian in 3 dimensions from 6-point stencil

    // Loop over stencil entries
    for(int j=0; j<2*ndim; j++)
    {
      SE = stencil.GetEntry(permute_type,j,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,Pnow->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = Pnow->_odata[SE->_offset];
      else
        SV = comm_buf[SE->_offset];
      if(j==0) {Lphi  = SV;}
      else     {Lphi += SV;}
    }

    // Complete Laplacian
    Lphi -= 6.0 * Pnow->_odata[i];
    return Lphi * dx2r;
  }

private:

  const int ndim   = 3;
  const int npoint = 6;
  const int Even   = 0;
  const std::vector<int> directions    = std::vector<int>({ 0, 1, 2, 0, 1, 2});
  const std::vector<int> displacements = std::vector<int>({ 1, 1, 1,-1,-1,-1});

  Grid::CartesianStencil stencil;
  std::vector< Walls::vScalar, Grid::alignedAllocator<Walls::vScalar> > comm_buf;
  Grid::SimpleCompressor<Walls::vScalar> compressor;

  const Grid::Real alfa       = 3.0;
  const Grid::Real beta0      = 0.0;
  const Grid::Real W0         = 10.0;

  const Grid::Real expoente   = 0.6666666;
  const Grid::Real ctinitial  = 1.0;
  const Grid::Real tampasso   = 0.25;

  const Grid::Real dct;
  const Grid::Real expoentec;
  const Grid::Real dx2r;
  Grid::Real ct;
  Grid::Real ct_prev;
  const Grid::Real omega0;
  const Grid::Real V0;

  Grid::Real delta;
  const Grid::Real idct;
  Grid::Real m1delta;
  Grid::Real V04atbeta0;
  const Grid::Real qmiu;
  Grid::Real i1pdelta;
};

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
    Walls::vScalarField P1(&Grid);
    Walls::vScalarField P2(&Grid);
    Walls::vScalarField * Pold;
    Walls::vScalarField * Pnow;
    Walls::vScalarField * Pnew;

    // Initialise scalar fields
    random(pRNG,P1);
    P2 = Grid::zero;
    Pold = NULL;
    Pnow = &P1;
    Pnew = &P2;

    // Check that P1 and P2 are consistent
    Grid::conformable(P1._grid,P2._grid);
    P2.checkerboard = P1.checkerboard;

    // Walls object
    Walls walls(&Grid);

    // Initial time step
    walls.initialise(Pnew,Pnow,Pold);

    // Switch array aliases
    Pold = Pnow;
    Pnow = Pnew;
    Pnew = Pold;

    // Start timer
    double start = Grid::usecond();

    // Loop over timesteps
    for(int ii=0; ii<Nloop; ii++)
    {
      // Initial time step
      walls.timestep(Pnew,Pnow,Pold);

      // Switch array aliases
      Pold = Pnow;
      Pnow = Pnew;
      Pnew = Pold;
    }

    // Total run time
    double stop = Grid::usecond();
    double time = (stop-start) * 1.0E-6;

    // Memory throughput and Flop rate
    double bytes = vol * sizeof(Grid::Real) * 2 * Nloop;
    double flops = double(vol) * double(20) * double(Nloop);
    std::cout << Grid::GridLogMessage << std::setprecision(3) << lat
              << "\t\t" << bytes << " \t\t" << 1.0E-9 * bytes / time
              << "\t\t" << 1.0E-9 * flops / time << std::endl;
  }

  Grid::Grid_finalize();
  return(0);
}
