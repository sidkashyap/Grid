/****************************************************************************/
/* pab: Signal magic. Processor state dump is x86-64 specific               */
/****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <sys/time.h>
#include <signal.h>
#include <iostream>
#include <iterator>
#include <Grid.h>
#include <algorithm>
#include <iterator>

#undef __X86_64
#define MAC

#ifdef MAC
#include <execinfo.h>
#endif

namespace Grid {

//////////////////////////////////////////////////////
// Convenience functions to access stadard command line arg
// driven parallelism controls
//////////////////////////////////////////////////////


static std::vector<int> Grid_default_latt;
static std::vector<int> Grid_default_mpi;
int GridThread::_threads;


const std::vector<int> &GridDefaultLatt(void)     {return Grid_default_latt;};
const std::vector<int> &GridDefaultMpi(void)      {return Grid_default_mpi;};




const std::vector<int> GridDefaultSimd(int dims,int nsimd)
{
    std::vector<int> layout(dims);
    int nn=nsimd;
    for(int d=dims-1;d>=0;d--){
      if ( nn>=2) {
	layout[d]=2;
	nn/=2;
      } else { 
	layout[d]=1;
      }
    }
    assert(nn==1);
    return layout;
}
  
////////////////////////////////////////////////////////////
// Command line parsing assist for stock controls
////////////////////////////////////////////////////////////


/**
 * returns the command line argument string which matches the string option.
 * returns null if there is no match
 *
 * Input:argv,argv+argc,"--mpi"/"--omp"/"--Grid" (based on the command line option
 * output: string payload or null string
 */

std::string GridCmdOptionPayload(char ** begin, char ** end, const std::string & option)
{
  //Returns an iterator to the first element in the range [first,last) that compares equal to val. If no such element is found, return last
  char ** itr = std::find(begin, end, option);

  if (itr != end && ++itr != end) {
    std::string payload(*itr);
    return payload;
  }
  return std::string("");
}
bool GridCmdOptionExists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}
  // Comma separated list
void GridCmdOptionCSL(std::string str,std::vector<std::string> & vec)
{
  size_t pos = 0;
  std::string token;
  std::string delimiter(",");

  vec.resize(0);
  while ((pos = str.find(delimiter)) != std::string::npos) {
    token = str.substr(0, pos);
    vec.push_back(token);
    str.erase(0, pos + delimiter.length());
  }
  token = str;
  vec.push_back(token);
  return;
}



/**
 *
 * Initialize the vector vec to the option specified by the user and override the defaults
 * the vector is usually specified with dots 4.4.4.4 etc, hence, remove the punctuation
 *
 * Input: arg, mpi/latt
 */

void GridCmdOptionIntVector(std::string &str,std::vector<int> & vec)
{
  vec.resize(0);
  std::stringstream ss(str);
  int i;
  while (ss >> i){
    vec.push_back(i);
    if(std::ispunct(ss.peek()))
      ss.ignore();
  }    
  return;
}
/**
 * Initializes the static vectors mpi and latt (declared in this namespace)
 * Input: Command line arguments and refernces to the vectors lattice and mpi
 * Output: Void, initializes several variables
 *
 */

//`./benchmarks/Grid_wilson --grid $vol --omp $omp  | grep mflop | awk '{print $3}'` echo $vol $perf >> wilson.t$omp

void GridParseLayout(char **argv,int argc,
		     std::vector<int> &latt,
		     std::vector<int> &mpi)
{
  mpi =std::vector<int>({1,1,1,1});
  latt=std::vector<int>({8,8,8,8});


  //set the static variable _threads defined in Threads.h to OMP_NUM_THREADS if openmp if not set _threads to 1
  GridThread::SetMaxThreads();

  std::string arg;

  if( GridCmdOptionExists(argv,argv+argc,"--mpi") ){
	//sets the string arg to the command line option that contains the string --mpi, if not returns null
    arg = GridCmdOptionPayload(argv,argv+argc,"--mpi");

    GridCmdOptionIntVector(arg,mpi);
  }

  //initialize the lattice vector to
  if( GridCmdOptionExists(argv,argv+argc,"--grid") ){
    arg= GridCmdOptionPayload(argv,argv+argc,"--grid");
    GridCmdOptionIntVector(arg,latt);
  }

  //if the commandline option is --omp, it should specify the number of threads only. parse the argument to find the number of threads and set the number of threads to be the same

  if( GridCmdOptionExists(argv,argv+argc,"--omp") ){
    std::vector<int> ompthreads(0);
    arg= GridCmdOptionPayload(argv,argv+argc,"--omp");
    GridCmdOptionIntVector(arg,ompthreads);
    assert(ompthreads.size()==1);
    GridThread::SetThreads(ompthreads[0]);
  }

}

std::string GridCmdVectorIntToString(const std::vector<int> & vec){
  std::ostringstream oss;
  std::copy(vec.begin(), vec.end(),std::ostream_iterator<int>(oss, " "));
  return oss.str();
}

/**
 * Initializes the requisite variables by parsing the command line arguments
 * Summary:
 *
 * Init MPI if the communication method chosen is MPI
 *
 * 1) Parse the command line arguments
 * 1.1) enable debug signals if asked
 * 1.2) choose the dslash and lebesque implementation as needed
 *
 * Inputs: command line arguments
 * outputs: void, but initializes several static and allocated variables across classes
 *
 * Invocation Example:
 *
 * omp in 1,2,4
 * vol in 4.4.4.4 4.4.4.8 4.4.8.8  4.8.8.8  8.8.8.8   8.8.8.16 8.8.16.16  8.16.16.16
 * ./benchmarks/Grid_wilson --grid $vol --omp $omp  | grep mflop | awk '{print $3}'`
 *
 */

void Grid_init(int *argc,char ***argv)
{
#ifdef GRID_COMMS_MPI
  MPI_Init(argc,argv);
#endif
  // Parse command line args.

  GridLogger::StopWatch.Start();

  std::string arg;
  std::vector<std::string> logstreams;
  std::string defaultLog("Error,Warning,Message,Performance");

  GridCmdOptionCSL(defaultLog,logstreams);
  GridLogConfigure(logstreams);

  if( GridCmdOptionExists(*argv,*argv+*argc,"--help") ){
    std::cout<<GridLogMessage<<"--help : this message"<<std::endl;
    std::cout<<GridLogMessage<<"--debug-signals : catch sigsegv and print a blame report"<<std::endl;
    std::cout<<GridLogMessage<<"--debug-stdout  : print stdout from EVERY node"<<std::endl;    
    std::cout<<GridLogMessage<<"--decomposition : report on default omp,mpi and simd decomposition"<<std::endl;    
    std::cout<<GridLogMessage<<"--mpi n.n.n.n   : default MPI decomposition"<<std::endl;    
    std::cout<<GridLogMessage<<"--omp n         : default number of OMP threads"<<std::endl;    
    std::cout<<GridLogMessage<<"--grid n.n.n.n  : default Grid size"<<std::endl;    
    std::cout<<GridLogMessage<<"--log list      : comma separted list of streams from Error,Warning,Message,Performance,Iterative,Debug"<<std::endl;    
  }

  if( GridCmdOptionExists(*argv,*argv+*argc,"--log") ){
    arg = GridCmdOptionPayload(*argv,*argv+*argc,"--log");
    GridCmdOptionCSL(arg,logstreams);
    GridLogConfigure(logstreams);
  }


  if( GridCmdOptionExists(*argv,*argv+*argc,"--debug-signals") ){
    Grid_debug_handler_init();
  }
  if( !GridCmdOptionExists(*argv,*argv+*argc,"--debug-stdout") ){
    Grid_quiesce_nodes();
  }
  if( GridCmdOptionExists(*argv,*argv+*argc,"--dslash-opt") ){
    QCD::WilsonFermionStatic::HandOptDslash=1;
  }
  if( GridCmdOptionExists(*argv,*argv+*argc,"--lebesgue") ){
    LebesgueOrder::UseLebesgueOrder=1;
  }


  //set the variables latt and mpi
  GridParseLayout(*argv,*argc,
		  Grid_default_latt,
		  Grid_default_mpi);



    if( GridCmdOptionExists(*argv,*argv+*argc,"--decomposition") ){
    std::cout<<GridLogMessage<<"Grid Decomposition\n";
    std::cout<<GridLogMessage<<"\tOpenMP threads : "<<GridThread::GetThreads()<<std::endl;
    std::cout<<GridLogMessage<<"\tMPI tasks      : "<<GridCmdVectorIntToString(GridDefaultMpi())<<std::endl;
    std::cout<<GridLogMessage<<"\tvRealF         : "<<sizeof(vRealF)*8    <<"bits ; " <<GridCmdVectorIntToString(GridDefaultSimd(4,vRealF::Nsimd()))<<std::endl;
    std::cout<<GridLogMessage<<"\tvRealD         : "<<sizeof(vRealD)*8    <<"bits ; " <<GridCmdVectorIntToString(GridDefaultSimd(4,vRealD::Nsimd()))<<std::endl;
    std::cout<<GridLogMessage<<"\tvComplexF      : "<<sizeof(vComplexF)*8 <<"bits ; " <<GridCmdVectorIntToString(GridDefaultSimd(4,vComplexF::Nsimd()))<<std::endl;
    std::cout<<GridLogMessage<<"\tvComplexD      : "<<sizeof(vComplexD)*8 <<"bits ; " <<GridCmdVectorIntToString(GridDefaultSimd(4,vComplexD::Nsimd()))<<std::endl;
  }


}

  
void Grid_finalize(void)
{
#ifdef GRID_COMMS_MPI
  MPI_Finalize();
  Grid_unquiesce_nodes();
#endif
}
double usecond(void) {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1.0*tv.tv_usec + 1.0e6*tv.tv_sec;
}

#define _NBACKTRACE (256)
void * Grid_backtrace_buffer[_NBACKTRACE];

void Grid_sa_signal_handler(int sig,siginfo_t *si,void * ptr)
{
  printf("Caught signal %d\n",si->si_signo);
  printf("  mem address %llx\n",(unsigned long long)si->si_addr);
  printf("         code %d\n",si->si_code);

#ifdef __X86_64
    ucontext_t * uc= (ucontext_t *)ptr;
  struct sigcontext *sc = (struct sigcontext *)&uc->uc_mcontext;
  printf("  instruction %llx\n",(unsigned long long)sc->rip);
#define REG(A)  printf("  %s %lx\n",#A,sc-> A);
  REG(rdi);
  REG(rsi);
  REG(rbp);
  REG(rbx);
  REG(rdx);
  REG(rax);
  REG(rcx);
  REG(rsp);
  REG(rip);


  REG(r8);
  REG(r9);
  REG(r10);
  REG(r11);
  REG(r12);
  REG(r13);
  REG(r14);
  REG(r15);
#endif
#ifdef MAC
  int symbols    = backtrace        (Grid_backtrace_buffer,_NBACKTRACE);
  char **strings = backtrace_symbols(Grid_backtrace_buffer,symbols);
  for (int i = 0; i < symbols; i++){
    printf ("%s\n", strings[i]);
  }
#endif
  exit(0);
  return;
};

void Grid_debug_handler_init(void)
{
  struct sigaction sa,osa;
  sigemptyset (&sa.sa_mask);
  sa.sa_sigaction= Grid_sa_signal_handler;
  sa.sa_flags    = SA_SIGINFO;
  sigaction(SIGSEGV,&sa,NULL);
  sigaction(SIGTRAP,&sa,NULL);
}
}
