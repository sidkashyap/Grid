#include <Grid.h>

class Walls
{

public:

  typedef Grid::iScalar<Grid::vRealD> vScalar;
  typedef Grid::Lattice<vScalar>      vScalarField;
  typedef vScalarField *              pvScalarField;
  typedef vScalarField const *        pcvScalarField;

  Walls(Grid::GridCartesian * Grid, vScalarField * P1, vScalarField * P2) :

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

    // Initialize pointers to scalar fields
    Pold = P2;
    Pnow = P1;
    Pnew = P2;

    // Initialize time varaibles
    ct_prev = ctinitial;
    ct      = ctinitial;
  }

  void timestep()
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
      vScalar Lphi = Laplacian(i);
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
    ct_prev  = ct;
    ct      += dct;

    // Switch array aliases
    Pold = const_cast<vScalarField const *>(Pnow);
    Pnow = const_cast<vScalarField const *>(Pnew);
    Pnew = const_cast<vScalarField *>(Pold);
  }

private:

  const int ndim   = 3;
  const int npoint = 6;
  const int Even   = 0;
  const std::vector<int> directions    = std::vector<int>({ 0, 1, 2, 0, 1, 2});
  const std::vector<int> displacements = std::vector<int>({ 1, 1, 1,-1,-1,-1});

  Grid::CartesianStencil stencil;
  std::vector< vScalar, Grid::alignedAllocator<vScalar> > comm_buf;
  Grid::SimpleCompressor<vScalar> compressor;

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

  vScalarField const * Pold;
  vScalarField const * Pnow;
  vScalarField *       Pnew;

  vScalar Laplacian(int i)
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
};

int main (int argc, char ** argv)
{
  // Initialize grid library
  Grid::Grid_init(&argc,&argv);

  // Benchmark constants
  const int Nloop = 100000; // Benchmark iterations
  const int Nd    = 3;      // Three spatial dimensions

  // Lattice, MPI and SIMD layouts and OpenMP threads
  std::vector<int> latt_size   = Grid::GridDefaultLatt();
  std::vector<int> mpi_layout  = Grid::GridDefaultMpi();
  std::vector<int> simd_layout = Grid::GridDefaultSimd(Nd,Grid::vRealD::Nsimd());
  int threads = Grid::GridThread::GetThreads();
  int vol     = latt_size[0]*latt_size[1]*latt_size[2];

  // Cartesian grid
  Grid::GridCartesian Grid(latt_size,simd_layout,mpi_layout);

  // Random number genereator
  Grid::GridParallelRNG pRNG(&Grid);
  pRNG.SeedRandomDevice();

  // Scalar fields P1 and P2 are initialized to the same sample
  // from the uniform random field [-1.0,1.0] on the Grid
  Walls::vScalarField P1(&Grid);
  Walls::vScalarField P2(&Grid);
  random(pRNG,P1);
  P1 = (2.*P1) - 1.;
  P2 = P1;

  // Walls object
  Walls walls(&Grid,&P1,&P2);

  // Start timer
  double start = Grid::usecond();

  // Loop over timesteps
  for(int ii=0; ii<Nloop; ii++) { walls.timestep(); }

  // Total run time
  double stop = Grid::usecond();
  double time = (stop-start) * 1.0E-6;

  // Memory throughput and Flop rate
  double bytes = vol * sizeof(Grid::Real) * 2 * Nloop;
  double flops = double(vol) * double(22) * double(Nloop);
  std::cout << Grid::GridLogMessage << std::setprecision(5)
            << "GB/s: "   << 1.0E-9 * bytes / time << "\t"
            << "Flop/s: " << 1.0E-9 * flops / time << std::endl;

  // Finalize
  Grid::Grid_finalize();
  return(0);
}
