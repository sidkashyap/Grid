#include <Grid.h>

int main (int argc, char ** argv)
{
  // Initialize grid library
  Grid::Grid_init(&argc,&argv);

  // Scalar field in Grid
  typedef Grid::Lattice< Grid::iScalar<Grid::vReal> > LatticeVec;

  int Nloop = 1000; // Benchmark iterations
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
  std::cout << Grid::GridLogMessage << "----------------------------------------------------------" <<std::endl;

  // Iterate over different local grid sizes given by lat.lat.lat
  for(int lat=4; lat<=32; lat+=4)
  {
    // Lattice size and volume
    std::vector<int> latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2]});
    int vol = latt_size[0]*latt_size[1]*latt_size[2];

    // Cartesian grid
    Grid::GridCartesian Grid(latt_size,simd_layout,mpi_layout);

    // RNG
    Grid::GridParallelRNG pRNG(&Grid);
    pRNG.SeedRandomDevice();

    // Random uniform scalar field P1
    LatticeVec P1(&Grid);
    random(pRNG,P1);

    // Scalar field Lphi of zeros
    LatticeVec Lphi(&Grid);
    Lphi = Grid::zero;

    // Start timer
    double start = Grid::usecond();

    // Iterate Nloop times for benchmark
    for(int i=0;i<Nloop;i++)
    {
      // 3D Laplacian
      Lphi = Grid::Cshift(P1,0,-1) + Grid::Cshift(P1,0,1)
           + Grid::Cshift(P1,1,-1) + Grid::Cshift(P1,1,1)
           + Grid::Cshift(P1,2,-1) + Grid::Cshift(P1,2,1)
           - 6.0 * P1;
    }

    // Total run tum
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
